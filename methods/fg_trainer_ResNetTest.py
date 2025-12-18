import torch
import socket
from datetime import datetime
from contextlib import contextmanager
from torch import nn

from helpers.trainer_class import BaseTrainer

class ForwardGradientTrainer_test(BaseTrainer):
    """Trainer-Klasse für Backpropagation-basiertes Training."""


    @contextmanager
    def disable_running_stats(self, model):
        # Finde alle BatchNorm Layer
        bns = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
        # Speichere den alten Zustand
        saved_states = [m.track_running_stats for m in bns]

        try:
            # Schalte Tracking aus -> BN nutzt Batch-Statistiken, schreibt aber nicht in Buffer
            for m in bns:
                m.track_running_stats = False
            yield
        finally:
            # Stelle alten Zustand wieder her
            for m, state in zip(bns, saved_states):
                m.track_running_stats = state

    def _train_epoch_impl(self):
        sum_loss = 0
        sum_correct = 0
        sum_size = 0

        pbar = self._create_progress_bar(desc=f'FGD- Train: {self.epoch_num}/{self.total_epochs}')
        named_params = dict(self.model.named_parameters())
        names = tuple(named_params.keys())
        named_buffers = dict(self.model.named_buffers())

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                self.model.train()

                # Warmup forward pass
                self.model(inputs)

                if self.should_track_memory_this_epoch:
                    if batch_idx < self.NUM_MEMORY_SNAPSHOTS_IN_EPOCH:
                        self.start_record_memory_history()

                buffers = {k: v.to(self.device) for k, v in named_buffers.items()}
                params = tuple(named_params.values())

                # Perturbation Vektor
                v_params = tuple(torch.randn_like(p) for p in params)

                # Loss function that returns (loss, output)
                def loss_fn(params_tuple):
                    params_dict = dict(zip(names, params_tuple))
                    state_dict = {**params_dict, **buffers}
                    output = torch.func.functional_call(self.model, state_dict, inputs)
                    loss = nn.functional.cross_entropy(output, targets)
                    return loss, output

                with self.disable_running_stats(self.model):
                    # JVP berechnet die Richtungsableitung
                    loss, outputs, dir_der = torch.func.jvp(
                        loss_fn,
                        (params,),
                        (v_params,),
                        has_aux=True
                    )

                sum_loss += loss.item()

                # Gradient Schätzung: grad ≈ directional_derivative * v (ohne Normierung)
                for j, param in enumerate(self.model.parameters()):
                    estimated_gradient = dir_der * v_params[j]
                    param.grad = estimated_gradient

                if batch_idx < self.NUM_MEMORY_SNAPSHOTS_IN_EPOCH:
                    self.export_memory_snapshot(batch_idx=batch_idx)
                    self.stop_record_memory_history()

                self.optimizer.step()

                # Metriken aktualisieren
                _, predicted = torch.max(outputs.data, 1)
                total = targets.size(0)
                correct = (predicted == targets).sum().item()
                sum_correct += correct
                sum_size += total
                accuracy = 100.0 * sum_correct / sum_size
                pbar.set_postfix({
                    'Loss': f'{loss:.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })

        self.metrics.loss_per_epoch = sum_loss / sum_size
        self.metrics.acc_per_epoch = 100. * sum_correct / sum_size
        self.metrics.num_batches = sum_size
