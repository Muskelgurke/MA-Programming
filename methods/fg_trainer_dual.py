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
        # Erst SubModul finden
        for part in parts[:-1]:
            obj = getattr(obj, part)

        attr_name = parts[-1]

        # Löschen WICHTIG!!!!!! Sonst Fehler
        if hasattr(obj, attr_name):
            delattr(obj, attr_name)

        # hinzufügen des DualTensors!
        setattr(obj, attr_name, value)

    @contextmanager
    def disable_running_stats(self, model):
        # BatchNorm Tracking temporär deaktivieren
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
        peak_memory = 0
        pbar = self._create_progress_bar(desc=f'FG Train: {self.epoch_num}')

        self.model.train()

        for batch_idx, (inputs, targets) in enumerate(pbar):
            if self.should_track_memory_this_epoch:
                if batch_idx < self.NUM_MEMORY_SNAPSHOTS_IN_EPOCH:
                    self.start_record_memory_history()

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            torch.cuda.reset_peak_memory_stats()

            #print(f"Mem before forward: {mem_pre_forward} bytes")
            mem_pre_forward = torch.cuda.memory_allocated(device=self.device)
            # BN Statistiken updaten - Original Parameter einsetzten
            for name, param in self.params_store.items():
                self._set_nested_attr(self.model, name, param)

            # Upadtes der Batchnormalization BUFFERS, sonst kein update mit Dual Tensoren.
            with torch.no_grad():
                self.model(inputs)

            with fwAD.dual_level():
                # Pertubation Vektor generieren für alle Parameter
                # und Dual Tensoren erstellen und ims Modell einsetztzen.
                v_dict = {}
                for name, param in self.params_store.items():
                    v = torch.randn_like(param)
                    v_dict[name] = v
                    dual_val = fwAD.make_dual(param, v)
                    self._set_nested_attr(self.model, name, dual_val)
                # Forward Pass
                # Dual Tensor CRASHES Gefahr. BatchNorm Running Stats deaktivieren
                with self.disable_running_stats(self.model):
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        loss = loss_func(outputs, targets)

                # Die Dual Tensoren auswerten in dem ich sie "UNPACKE" -> heraushole aus dem modell.
                dual_loss = fwAD.unpack_dual(loss)
                loss_val = dual_loss.primal
                jvp = dual_loss.tangent  # Gradient Richtung

                if jvp is None or torch.isnan(jvp):
                    # wiederherstellung der originalen Parameter
                    for name, param in self.params_store.items():
                        self._set_nested_attr(self.model, name, param)
                    continue

            #Gradienten werden berechnet über den wFGD TRICK und dann in das Modell gesetzt
            with torch.no_grad():
                for name, param in self.params_store.items():
                    v = v_dict[name]
                    # Update Regel: grad = (dL/dv) * v
                    if param.grad is None:
                        param.grad = jvp * v
                    else:
                        param.grad += jvp * v

            # Forward Pass Memory messen.
            mem_post_forward = torch.cuda.memory_allocated(device=self.device)
            mem_forward.append(mem_post_forward - mem_pre_forward)
            #print(f"Mem after forward: {mem_post_forward} bytes, used: {mem_post_forward - mem_pre_forward} bytes")

            # Da immernoch dual tensoren drin sind im Modell
            # müssen diese wieder rausgenommen werden und mit den originalen Parametern ersetzt werden.

            for name, param in self.params_store.items():
                self._set_nested_attr(self.model, name, param)

            self.optimizer.step()
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()

            sum_loss += loss_val.item() * inputs.size(0)

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
        self.metrics.peak_mem_bytes = peak_memory

        self.metrics.loss_per_epoch = sum_loss / sum_size
        self.metrics.acc_per_epoch = 100. * sum_correct / sum_size
        self.metrics.num_batches = sum_size