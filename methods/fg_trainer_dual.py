import torch
import torch.autograd.forward_ad as fwAD
import numpy as np
from torch import nn
from helpers.trainer_class import BaseTrainer
from contextlib import contextmanager

class ForwardGradientTrainer_dual(BaseTrainer):
    """
    Implementiert Forward Gradient Descent basierend auf der Logik des FROG-Repos.
    Trick: Ersetzt nn.Parameter durch Dual-Tensors direkt im Modell.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 1. HACK: Parameter "befreien"
        # Wir speichern die echten Parameter separat und löschen sie aus der Modell-Registrierung,
        # damit wir sie später durch Dual Tensors ersetzen können.
        self.params_store = {}
        self.param_names = []

        # Wir müssen über named_parameters iterieren und sie "detachen"
        for name, param in self.model.named_parameters():
            self.params_store[name] = param  # Referenz behalten für Optimizer
            self.param_names.append(name)

        # Jetzt entfernen wir die Parameter aus dem Modell-Objekt
        # Das macht Platz für unsere Dual Tensors
        for name in self.param_names:
            self._del_nested_attr(self.model, name)

        # WICHTIG: Der Optimizer muss VORHER initialisiert worden sein mit den originalen Parametern!
        # Da BaseTrainer den Optimizer meist im __init__ erstellt, passt das.
        # Aber Vorsicht: model.parameters() ist jetzt leer!

    def _del_nested_attr(self, obj, name):
        """Hilfsfunktion um Attribute wie 'layer1.weight' zu löschen"""
        parts = name.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        delattr(obj, parts[-1])

    def _set_nested_attr(self, obj, name, value):
        """
        Setzt ein Attribut und löscht es vorher, um Konflikte zwischen
        nn.Parameter und regulären Tensors/DualTensors zu vermeiden.
        """
        parts = name.split('.')
        # Zum richtigen Submodul navigieren (z.B. layer1.conv1)
        for part in parts[:-1]:
            obj = getattr(obj, part)

        attr_name = parts[-1]

        # WICHTIG: Erst löschen! Das entfernt 'weight' aus self._parameters
        if hasattr(obj, attr_name):
            delattr(obj, attr_name)

        # Jetzt neu setzen (als Tensor, DualTensor oder Parameter)
        setattr(obj, attr_name, value)

    @contextmanager
    def disable_running_stats(self, model):
        # BN Fix wie gehabt
        bns = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
        saved_states = [m.track_running_stats for m in bns]
        try:
            for m in bns: m.track_running_stats = False
            yield
        finally:
            for m, state in zip(bns, saved_states): m.track_running_stats = state

    def _train_epoch_impl(self):
        sum_loss = 0
        sum_correct = 0
        sum_size = 0
        loss_func = nn.CrossEntropyLoss()
        mem_pre_forward = 0
        mem_forward = []

        pbar = self._create_progress_bar(desc=f'FG Train: {self.epoch_num}')

        self.model.train()

        for batch_idx, (inputs, targets) in enumerate(pbar):
            if self.should_track_memory_this_epoch:
                if batch_idx < self.NUM_MEMORY_SNAPSHOTS_IN_EPOCH:
                    self.start_record_memory_history()

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            if self.should_track_memory_this_epoch:
                mem_pre_forward = torch.cuda.memory_allocated(device=self.device)
                #print(f"Mem before forward: {mem_pre_forward} bytes")

            # temporär die echten Tensors (keine Duals) ein,
            # damit BatchNorm Statistiken sammeln kann.
            for name, param in self.params_store.items():
                self._set_nested_attr(self.model, name, param)

            # Warmup Forward Pass (ohne Gradienten, rein für BatchNorm Statistiken)
            with torch.no_grad():
                self.model(inputs)


            with fwAD.dual_level():

                # 1. Perturbation generieren & Dual Tensors setzen
                v_dict = {}
                for name, param in self.params_store.items():

                    v = torch.randn_like(param)
                    v_dict[name] = v

                    dual_val = fwAD.make_dual(param, v)

                    self._set_nested_attr(self.model, name, dual_val)

                # Forward Pass
                # Wir deaktivieren BN-Tracking, da wir nicht in die Buffer schreiben wollen (Dual Tensor Crash Gefahr)
                with self.disable_running_stats(self.model):
                    outputs = self.model(inputs)
                    loss = loss_func(outputs, targets)

                # Unpack dual Tensors
                dual_loss = fwAD.unpack_dual(loss)
                loss_val = dual_loss.primal
                jvp = dual_loss.tangent  # Gradient Richtung


                if jvp is None or torch.isnan(jvp):
                    # wiederherstellung der originalen Parameter
                    for name, param in self.params_store.items():
                        self._set_nested_attr(self.model, name, param)
                    continue

            #Gradient setzen
            with torch.no_grad():
                for name, param in self.params_store.items():
                    v = v_dict[name]
                    # Update Regel: grad = (dL/dv) * v
                    if param.grad is None:
                        param.grad = jvp * v
                    else:
                        param.grad += jvp * v  # Falls man Batch Accumulation macht

            if self.should_track_memory_this_epoch:
                mem_post_forward = torch.cuda.memory_allocated(device=self.device)
                mem_forward.append(mem_post_forward - mem_pre_forward)
                #print(f"Mem after forward: {mem_post_forward} bytes, used: {mem_post_forward - mem_pre_forward} bytes")

            # WICHTIG: Vor dem Optimizer Step müssen wir die Dual Tensors wieder
            # durch die echten Parameter ersetzen!
            for name, param in self.params_store.items():
                self._set_nested_attr(self.model, name, param)

            self.optimizer.step()


            sum_loss += loss_val.item() * inputs.size(0)

            # Primal Output für Accuracy nutzen
            primal_output = fwAD.unpack_dual(outputs).primal
            _, predicted = torch.max(primal_output, 1)
            sum_correct += (predicted == targets).sum().item()
            sum_size += targets.size(0)

            acc = 100.0 * sum_correct / sum_size

            pbar.set_postfix({
                'Loss': f'{loss_val.item():.4f}',
                'ACC': f'{acc:.4f}'
            })


        self.metrics.avg_mem_forward_pass_bytes = int(np.mean(mem_forward)) if mem_forward else 0
        self.metrics.max_mem_forward_pass_bytes = int(np.max(mem_forward)) if mem_forward else 0

        self.metrics.loss_per_epoch = sum_loss / sum_size
        self.metrics.acc_per_epoch = 100. * sum_correct / sum_size
        self.metrics.num_batches = sum_size