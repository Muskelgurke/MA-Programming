import torch
import numpy as np
from helpers.trainer_class import BaseTrainer


class BackpropTrainer(BaseTrainer):
    """Trainer-Klasse für Backpropagation-basiertes Training."""

    def _train_epoch_impl(self):
        sum_loss = 0
        sum_correct = 0
        sum_size = 0
        pbar = self._create_progress_bar(desc=f'BP - Train: {self.epoch_num}/{self.total_epochs}')

        mem_pre_forward = 0
        mem_post_forward = 0
        mem_forward = []
        mem_backward = []
        """
        prof_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)
        with profile(
            activities= [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule,
            profile_memory=True,
            record_shapes=True,
            on_trace_ready= trace_handler,
            with_stack=True
        )as prof:
        """

        for batch_idx, (inputs, targets) in enumerate(pbar):

            inputs, targets = inputs.to(device=self.device), targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            if self.should_track_memory_this_epoch:
                mem_pre_forward = torch.cuda.memory_allocated(device=self.device)
                if batch_idx < self.NUM_MEMORY_SNAPSHOTS_IN_EPOCH:
                    self.start_record_memory_history()

            outputs = self.model(inputs)

            if self.should_track_memory_this_epoch:
                mem_post_forward = torch.cuda.memory_allocated(device=self.device)
                mem_forward.append(mem_post_forward - mem_pre_forward)

            loss = self.loss_function(outputs, targets)
            sum_loss += loss.item()

            loss.backward()

            if self.should_track_memory_this_epoch:

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

            # Speicher-Metriken aktualisieren
            # memory_activiations are saved in the first batch

        self.metrics.avg_mem_forward_pass_bytes = int(np.mean(mem_forward)) if mem_forward else 0
        self.metrics.max_mem_forward_pass_bytes = int(np.max(mem_forward)) if mem_forward else 0
        # Epoche-Metriken aktualisieren
        self.metrics.loss_per_epoch = sum_loss / sum_size
        self.metrics.acc_per_epoch = 100. * sum_correct / sum_size
        self.metrics.num_batches = sum_size