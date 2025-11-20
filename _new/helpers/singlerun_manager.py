# singlerun_manager.py

import torch
import time
import statistics
import gc



# Importiere Trainer, Tester, EarlyStopping, TorchModelSaver, etc.

class SingleRunManager:
    """Verwaltet einen einzelnen Trainingslauf von A bis Z."""

    def __init__(self, config: Config, run_number: int, device: torch.device):
        self.config = config
        self.run_number = run_number
        self.device = device
        self.start_time = time.time()

        # Initialisierung von Ladern, Modell, Optimierer, Logger, Saver, Early Stopping
        self._setup_run()

    def _setup_run(self):
        # Lade Daten, Modell, Optimierer, Loss-Funktion (wie in get_optimizer_and_lossfunction)
        self.train_loader, self.test_loader = datasets_helper.get_dataloaders(self.config, self.device)
        self.model = model_helper.get_model(self.config).to(self.device)
        self.loss_function, self.optimizer = get_optimizer_and_lossfunction(self.config, self.model)

        # Setup Logging und Saving
        timestamp = datetime.datetime.now().strftime("%H_%M_%S")
        today = datetime.datetime.now().strftime("%Y_%m_%d")
        run_path = f'runs/{today}/{timestamp}_run{self.run_number}_...'  # Kürze den Pfad
        self.writer = SummaryWriter(log_dir=run_path)
        self.saver = TorchModelSaver(self.writer.log_dir)

        # Setup Early Stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            delta=self.config.early_stopping_delta
        ) if self.config.early_stopping else None

        # Initialisiere Trainer und Tester
        self.trainer = Trainer(
            config_file=self.config, model=self.model, data_loader=self.train_loader,
            loss_function=self.loss_function, optimizer=self.optimizer,
            device=self.device, total_epochs=self.config.epoch_num,
            seed=self.config.random_seed, tensorboard_writer=self.writer,
            saver_class=self.saver, early_stopping_class=self.early_stopping
        )
        self.tester = Tester(...)  # Fülle die Parameter für Tester aus

    def run(self) -> dict:
        """Führt die Haupt-Epochen-Schleife durch und gibt die Ergebnisse zurück."""
        train_losses_per_epoch = []
        test_accs_per_epoch = []
        # ... weitere Listen für Metriken ...

        try:
            for epoch in range(self.config.epoch_num):
                start_epoch_time = time.time()

                # 1. Training
                training_results = self.trainer.train_epoch(epoch_num=epoch)
                # Prüfe auf NaN-Loss (Logik aus main_start_session.py und trainer.py)
                if self.early_stopping and self.early_stopping.early_stop_nan_train_loss:
                    self.save_early_stop_summary(training_results, test_accs_per_epoch, epoch)
                    break

                # 2. Testing
                tester_results = self.tester.validate_epoch(epoch_num=epoch)

                # 3. Speichern und Logging
                self.saver.write_epoch_metrics(...)
                # Füge Metriken zu Listen hinzu
                train_losses_per_epoch.append(training_results.epoch_avg_train_loss)
                test_accs_per_epoch.append(tester_results.test_acc_per_epoch)

                # 4. Early Stopping Check
                if self.early_stopping:
                    self.early_stopping.check_validation(tester_results.test_loss_per_epoch, self.model)
                    if self.early_stopping.early_stop:
                        # Schreibe Early-Stop-Zusammenfassung und beende die Schleife
                        self.save_early_stop_summary(training_results, test_accs_per_epoch, epoch)
                        break

            # 5. Finale Speicherung
            self.saver.save_session_after_epoch(config=self.config, model=self.model, save_full_model=True)
            return self._finalize_results(train_losses_per_epoch, test_accs_per_epoch)

        except Exception as e:
            # Fehlerbehandlung
            raise e
        finally:
            self._cleanup()

    def _save_early_stop_summary(self, training_results, test_accs_per_epoch, epoch):
        run_time = time.time() - self.start_time
        # Die Logik zum Schreiben des run_summary hier einfügen
        # ...

    def _finalize_results(self, train_losses, test_accs):
        # Erstelle das endgültige Ergebnis-Dict
        return {
            'final_train_acc': train_losses[-1] if train_losses else 0.0,
            # ...
        }

    def _cleanup(self):
        # Logik zum Aufräumen von DataLoadern, Modell, GPU-Speicher (aus main_start_session.py)
        if self.writer:
            self.writer.flush()
            self.writer.close()
        # ... (restliches Cleanup) ...
        gc.collect()