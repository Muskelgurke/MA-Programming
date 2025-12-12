
import torch
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from helpers.trainer_class import BaseTrainer


class BackpropTrainer(BaseTrainer):
    """Trainer-Klasse für Backpropagation-basiertes Training."""

    def _train_epoch_impl(self):
        sum_loss = 0
        sum_correct = 0
        sum_size = 0
        to_mb = (1024 ** 2)
        pbar = self._create_progress_bar(desc=f'BP - Train: {self.epoch_num}/{self.total_epochs}')

        prof_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)
        with profile(
            activities= [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule,
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


        if prof:
            print("Profiling ergebnisse für Epoche")
            # Hol alle Events
            events = prof.key_averages()

            # Suche effizient nach deinen Labels
            evt_dict = {e.key: e for e in events}
            print("Profiling events in dict umgewandelt")
            if "bp_forward" in evt_dict:
                evt = evt_dict["bp_forward"]
                # WICHTIG: Nimm 'cuda_memory_usage' (inkl. Kinder), nicht 'self_'
                memory_bytes = getattr(evt, "cuda_memory_usage", 0)
                time_us = getattr(evt, "cuda_time_total", 0)

                # Umrechnung für bessere Lesbarkeit
                memory_mb = memory_bytes / (1024 ** 2)
                time_ms = time_us / 1000.0

                print(f"--- DEBUG bp_forward ---")
                print(f"Memory (Total): {memory_mb:.2f} MB")
                print(f"Time (Total):   {time_ms:.2f} ms")
                #self.metrics.memory_forward_pass_MB = evt_dict["bp_forward"].self_cuda_memory_usage / to_mb
                print("Profiling forward pass memory geschrieben")

            if "bp_backward" in evt_dict:
                evt = evt_dict["bp_forward"]
                # WICHTIG: Nimm 'cuda_memory_usage' (inkl. Kinder), nicht 'self_'
                memory_bytes = getattr(evt, "cuda_memory_usage", 0)
                time_us = getattr(evt, "cuda_time_total", 0)

                # Umrechnung für bessere Lesbarkeit
                memory_mb = memory_bytes / (1024 ** 2)
                time_ms = time_us / 1000.0

                print(f"--- DEBUG bp_backward ---")
                print(f"Memory (Total): {memory_mb:.2f} MB")
                print(f"Time (Total):   {time_ms:.2f} ms")
                #self.metrics.memory_backward_pass_MB = evt_dict["bp_backward"].self_cuda_memory_usage / to_mb
                print("Profiling backward pass memory geschrieben")

        # Speicher-Metriken aktualisieren
        # memory_activiations are saved in the first batch
        self.metrics.memory_peak_MB = torch.cuda.max_memory_allocated(self.device) / to_mb

        # Epoche-Metriken aktualisieren
        self.metrics.loss_per_epoch = sum_loss / sum_size
        self.metrics.acc_per_epoch = 100. * sum_correct / sum_size
        self.metrics.num_batches = sum_size