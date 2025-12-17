import socket
from datetime import datetime, timedelta
import torch
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from helpers.trainer_class import BaseTrainer


class BackpropTrainer(BaseTrainer):
    """Trainer-Klasse für Backpropagation-basiertes Training."""

    def _train_epoch_impl(self):
        self.MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 10000
        self.TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
        self.NUM_MEMORY_SNAPSHOTS: int = 3

        def start_record_memory_history() -> None:
            print("Starte Speicher-Verlaufsaufzeichnung...")
            torch.cuda.memory._record_memory_history(max_entries=self.MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)

        def stop_record_memory_history() -> None:
            print("Stoppe Speicher-Verlaufsaufzeichnung...")
            torch.cuda.memory._record_memory_history(enabled=None)

        def export_memory_snapshot() -> None:
            # Prefix for file names.
            host_name = socket.gethostname()
            timestamp = datetime.now().strftime(self.TIME_FORMAT_STR)
            file_name = f"{host_name}_{timestamp}"
            file_path = f"{self.runsPath}/{file_name}"

            try:
                print(f"Saving snapshot to local file: {file_path}.pickl")
                torch.cuda.memory._dump_snapshot(f"{file_path}.pickle")
            except Exception as e:
                print(f"Failed to capture memory snapshot {e}")
                return

        sum_loss = 0
        sum_correct = 0
        sum_size = 0
        pbar = self._create_progress_bar(desc=f'BP - Train: {self.epoch_num}/{self.total_epochs}')

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
            self.optimizer.zero_grad()

            if batch_idx < self.NUM_MEMORY_SNAPSHOTS:
                start_record_memory_history()

            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            sum_loss += loss.item()

            loss.backward()

            if batch_idx < self.NUM_MEMORY_SNAPSHOTS:
                export_memory_snapshot()
                stop_record_memory_history()

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


            # Epoche-Metriken aktualisieren
            self.metrics.loss_per_epoch = sum_loss / sum_size
            self.metrics.acc_per_epoch = 100. * sum_correct / sum_size
            self.metrics.num_batches = sum_size