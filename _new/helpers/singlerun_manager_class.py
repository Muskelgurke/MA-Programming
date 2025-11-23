import torch
import time
import datetime
from _new.helpers.saver_class import TorchModelSaver
from _new.helpers.config_class import Config
from _new.helpers.tester_class import Tester
from _new.helpers.trainer_class import BaseTrainer
from _new.methods.backprop_trainer import BackpropTrainer
from _new.methods.fg_trainer import ForwardGradientTrainer
from _new.helpers.early_stopping_class import EarlyStopping

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

        self.tester = self._create_tester()  # Fülle die Parameter für Tester aus

        self.model = None

        # Initialize early stopping at the manager level
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            delta=self.config.early_stopping_delta
        ) if self.config.early_stopping else None

    def _create_path(self) -> str:
        today = datetime.datetime.now().strftime("%Y_%m_%d")
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_path = f'runs/start{today}/run{self.run_number}_time{timestamp}_{self.config.dataset_name}_{self.config.model_type}_{self.config.training_method}'
        return run_path

    def _create_tester(self):
        return Tester(config_file=self.config,
                      device=self.device,
                      saver_class=self.saver)

    def _create_trainer(self) -> BaseTrainer:

        match self.config.training_method:
            case "fgd":
                return ForwardGradientTrainer(config_file=self.config,
                                              device=self.device,
                                              saver_class=self.saver)
            case "bp":
                return BackpropTrainer(config_file=self.config,
                                       device=self.device,
                                       saver_class=self.saver)
            case _:
                raise ValueError(f"Unknown Training - Method: {self.config.training_method}")

    def run(self) -> dict:
        """Führt die Haupt-Epochen-Schleife durch und gibt die Ergebnisse zurück."""
        try:
            for epoch in range(self.config.epoch_total):
                start_epoch_time = time.time()

                train_metrics, self.model = self.trainer.train_epoch(epoch_num=epoch)

                test_metrics = self.tester.validate_epoch(epoch_num=epoch)

                self.saver.write_epoch_metrics(train_metrics=train_metrics,
                                               test_metrics=test_metrics,
                                               epoch_idx=epoch)

                if self.early_stopping:
                    self.early_stopping.check_and_update(
                        train_loss=train_metrics.loss_per_epoch,
                        val_loss=test_metrics.loss_per_epoch,
                        model=self.trainer.model,
                        epoch=epoch
                    )
                    if self.early_stopping.early_stop:
                        break

            self.saver.save_session_after_epochs(config=self.config,
                                                 model=self.model ,
                                                 save_full_model=True,
                                                 stop_info= self.early_stopping.get_stop_info() if self.early_stopping else None)

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