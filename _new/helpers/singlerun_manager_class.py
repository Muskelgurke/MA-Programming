import torch
import time
import datetime
from _new.helpers.saver_class import TorchModelSaver
from _new.helpers.config_class import Config
from _new.trainer.trainer_class import BaseTrainer
from _new.tester.tester_class import Tester

class SingleRunManager:
    """Verwaltet einen einzelnen Trainingslauf von A bis Z."""

    def __init__(self, config: Config, run_number: int, device: torch.device):
        self.config = config
        self.run_number = run_number
        self.device = device
        self.start_time = time.time()

        self._setup_run()

    def _setup_run(self):
        # Saver Setup
        run_path = self._create_path()

        self.saver = TorchModelSaver(run_path)

        self.trainer = self._create_trainer()

        self.tester = Tester(...)  # Fülle die Parameter für Tester aus

    def _create_path(self) -> str:
        today = datetime.datetime.now().strftime("%Y_%m_%d")
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_path = f'runs/start{today}/run{self.run_number}_time{timestamp}_{self.config.dataset_name}_{self.config.model_type}_{self.config.training_method}'
        return run_path
    def _create_trainer(self) -> BaseTrainer:

        match self.config.training_method:
            case "fgd":
                return ForwardGradientTrainer(config=self.config,
                                              device=self.device,
                                              saver=self.saver)
            case "bp":
                return BackpropagationTrainer(config=self.config,
                                              device=self.device,
                                              saver=self.saver)
            case _:
                raise ValueError(f"Unknown Training - Method: {self.config.training_method}")

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