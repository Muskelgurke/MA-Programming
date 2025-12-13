import socket
from datetime import datetime, timedelta
import torch
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from helpers.trainer_class import BaseTrainer


class BackpropTrainer(BaseTrainer):
    """Trainer-Klasse für Backpropagation-basiertes Training."""
    TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
    def _train_epoch_impl(self):
        def trace_handler(prof: torch.profiler.profile):
            # Prefix for file names.
            TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

            host_name = socket.gethostname()
            timestamp = datetime.now().strftime(TIME_FORMAT_STR)
            file_prefix = f"{host_name}_{timestamp}"

            # Construct the trace file.
            prof.export_chrome_trace(f"{file_prefix}.json.gz")

            # Construct the memory timeline file.
            prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
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
            on_trace_ready= trace_handler
        )as prof:

            for batch_idx, (inputs, targets) in enumerate(pbar):
                prof.step()
                inputs, targets = inputs.to(device=self.device,non_blocking=True), targets.to(self.device,non_blocking=True)
                self.optimizer.zero_grad()
                if batch_idx == 5:
                    torch.cuda.empty_cache()

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
                # Profiling Schritt abschließen

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

        exit()
        print("\n" + "=" * 50)
        print("PROFILER DIAGNOSE: bp_forward")
        print("=" * 50)
        #events_test = prof.
        events = prof.key_averages()
        evt_dict = {e.key: e for e in events}

        if "bp_forward" in evt_dict:
            evt = evt_dict["bp_forward"]

            # Umrechnungsfaktoren

            # --- 1. Die Standard-Zusammenfassung von PyTorch ---
            print("--- PyTorch Roh-Daten (String Repr) ---")
            print(evt)  # Ruft automatisch die __str__ Methode auf, zeigt oft schon viel an.
            print("-" * 30)

            # --- 2. Manuelle Aufschlüsselung aller Werte ---
            print(f"Anzahl Aufrufe (Count): {evt.count}")

            # ZEIT (Time)
            # Total = Die Funktion + alles was sie aufgerufen hat (Conv2d, ReLU, etc.)
            # Self  = Nur der Python-Wrapper Overhead selbst
            print("\n[ZEIT - TIME]")
            print(f"Self CPU Time:       {evt.self_cpu_time_total :.4f} ms")
            print(f"Self CUDA Time:      {evt.self_cuda_time :.4f} ms")

            # SPEICHER (Memory)
            print("\n[SPEICHER - MEMORY]")
            # Achtung: Zeigt oft 0.00 MB, wenn PyTorch den Speicher aus dem Cache nimmt!
            print(f"CPU Memory Usage:    {evt.cpu_memory_usage } MB")
            print(f"CUDA Memory Usage:   {evt.cuda_memory_usage } MB  <-- Dein gesuchter Wert")

            # FORM (Shapes) - Falls record_shapes=True aktiviert war
            print("\n[INPUT SHAPES]")
            print(f"Input Shapes:        {evt.input_shapes}")

        else:
            print("WARNUNG: 'bp_forward' wurde im Profiler nicht gefunden!")
            print("Gefundene Keys:", list(evt_dict.keys())[:5], "...")  # Zeige die ersten 5 Keys zur Hilfe

        print("=" * 50 + "\n")

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