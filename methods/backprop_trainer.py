
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

        print("\n" + "=" * 50)
        print("PROFILER DIAGNOSE: bp_forward")
        print("=" * 50)

        events = prof.key_averages()
        evt_dict = {e.key: e for e in events}

        if "bp_forward" in evt_dict:
            evt = evt_dict["bp_forward"]

            # Umrechnungsfaktoren
            to_mb = 1024 ** 2
            to_ms = 1000.0

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
            print(f"CPU Time Total:      {evt.cpu_time_total / to_ms:.4f} ms")
            print(f"CUDA Time Total:     {evt.cuda_time_total / to_ms:.4f} ms  <-- Dein gesuchter Wert für den Graph")
            print(f"Self CPU Time:       {evt.self_cpu_time_total / to_ms:.4f} ms")
            print(f"Self CUDA Time:      {evt.self_cuda_time_total / to_ms:.4f} ms")

            # SPEICHER (Memory)
            print("\n[SPEICHER - MEMORY]")
            # Achtung: Zeigt oft 0.00 MB, wenn PyTorch den Speicher aus dem Cache nimmt!
            print(f"CPU Memory Usage:    {evt.cpu_memory_usage / to_mb:.4f} MB")
            print(f"CUDA Memory Usage:   {evt.cuda_memory_usage / to_mb:.4f} MB  <-- Dein gesuchter Wert")
            print(f"Self CPU Memory:     {evt.self_cpu_memory_usage / to_mb:.4f} MB")
            print(f"Self CUDA Memory:    {evt.self_cuda_memory_usage / to_mb:.4f} MB")

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