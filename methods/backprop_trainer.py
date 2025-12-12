from helpers.trainer_class import BaseTrainer
import torch
from torch.profiler import profile, ProfilerActivity, record_function



class BackpropTrainer(BaseTrainer):
    """Trainer-Klasse für Backpropagation-basiertes Training."""

    def _train_epoch_impl(self):
        sum_loss = 0
        sum_correct = 0
        sum_size = 0

        pbar = self._create_progress_bar(desc=f'BP - Train: {self.epoch_num}/{self.total_epochs}')

        with profile(
            activities= [ProfilerActivity.CPU, ProfilerActivity.GPU],
            profile_memory=True,
            record_shapes=True,
        )as prof:

            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                mem_start = torch.cuda.memory_allocated(self.device)
                with record_function("bp_forward"):
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, targets)
                # Speicher  (Gewichte + Outputs + Graph)
                mem_after_forward = torch.cuda.memory_allocated(self.device)

                sum_loss += loss.item()

                # Backward Pass
                with record_function("bp_backward"):
                    loss.backward()
                # Speicher  (Gewichte + Gradients + Outputs - Graph)
                mem_after_backward = torch.cuda.memory_allocated(self.device)

                self.optimizer.step()
                prof.step() # Profiling Schritt abschließen

                # Metriken aktualisieren
                _, predicted = torch.max(outputs.data, 1)
                total = targets.size(0)
                correct = (predicted == targets).sum().item()

                if batch_idx == 0:
                    mem_activations = (mem_after_forward - mem_start) / (1024 ** 2)
                    self.metrics.memory_activations_MB = mem_activations

                sum_correct += correct
                sum_size += total
                accuracy = 100.0 * sum_correct / sum_size

                pbar.set_postfix({
                    'Loss': f'{loss:.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })


        # Speicher-Metriken aktualisieren
        # memory_activiations are saved in the first batch
        self.metrics.memory_forward_pass_MB = self.extract_profiler_metric(prof, 'bp_forward')
        self.metrics.memory_backward_pass_MB = self.extract_profiler_metric(prof, 'bp_backward')
        self.metrics.memory_peak_MB = torch.cuda.max_memory_allocated(self.device) / (1024**2)

        # Epoche-Metriken aktualisieren
        self.metrics.loss_per_epoch = sum_loss / sum_size
        self.metrics.acc_per_epoch = 100. * sum_correct / sum_size
        self.metrics.num_batches = sum_size