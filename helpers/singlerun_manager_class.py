import torch
import time
import datetime
from pathlib import Path
from dataclasses import replace
from helpers.saver_class import TorchModelSaver
from helpers.config_class import Config
from helpers.validation_class import Tester
from helpers.trainer_class import BaseTrainer
from methods.backprop_trainer import BackpropTrainer
from methods.fg_trainer import ForwardGradientTrainer
from methods.fg_trainer_ResNetTest import ForwardGradientTrainer_test
from methods.fg_trainer_dual import ForwardGradientTrainer_dual
from helpers.early_stopping_class import EarlyStopping

class SingleRunManager:
    """Verwaltet einen einzelnen Trainingslauf von A bis Z."""

    def __init__(self, config: Config, run_number: int, device: torch.device, base_path: str):
        self.config = config
        self.run_number = run_number
        self.device = device
        self.total_time = 0
        self.base_path = base_path

        self._setup_run()

    def _setup_run(self):
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        run_name = f'run{self.run_number}_d{self.config.cuda_device}_t{timestamp}_{self.config.dataset_name}_{self.config.model_type}_{self.config.training_method}'
        run_path = Path(self.base_path) / run_name

        self.saver = TorchModelSaver(base_path=self.base_path,
                                     run_path=run_path)

        self.trainer = self._create_trainer()

        self.tester = self._create_tester()

        self.trained_model = None

        # Initialize early stopping at the manager level
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            delta=self.config.early_stopping_delta,
            max_loss_threshold= 10000
        ) if self.config.early_stopping else None

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
            case "fgd_test":
                return ForwardGradientTrainer_test(config_file=self.config,
                                              device=self.device,
                                              saver_class=self.saver)
            case "fgd_dual":
                return ForwardGradientTrainer_dual(config_file=self.config,
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
            total_start_time = time.time()
            train_metrics = None
            test_metrics = None
            best_test_metrics = None
            best_train_metrics = None
            best_test_acc = -float('inf')
            convergence_epoch = -1
            epoch_durations = []

            for epoch in range(self.config.epoch_total):

                train_metrics, self.trained_model = self.trainer.train_epoch(epoch_num=epoch)
                print("Training beendet. Starte Validierung...")
                test_metrics = self.tester.validate_epoch(epoch_num=epoch, model = self.trained_model)
                print("Validierung beendet. Write Epoch csv...")
                if hasattr(train_metrics, 'epoch_duration'):
                    epoch_durations.append(train_metrics.epoch_duration)

                self.saver.write_epoch_metrics_csv(train_metrics=train_metrics,
                                                   test_metrics=test_metrics,
                                                   epoch_idx=epoch)

                print("update best metrics and check early stopping...")

                if test_metrics.acc_per_epoch > best_test_acc:
                    best_test_acc = test_metrics.acc_per_epoch
                    best_test_metrics = replace(test_metrics)
                    best_train_metrics = replace(train_metrics)
                    convergence_epoch = epoch

                if self.early_stopping:
                    self.early_stopping.check_and_update(
                        train_loss=train_metrics.loss_per_epoch,
                        val_loss=test_metrics.loss_per_epoch,
                        model=self.trained_model,
                        epoch=epoch
                    )
                    print(f"Early stopping counter: {self.early_stopping.counter} / {self.early_stopping.patience}")
                    print(f"")
                    if self.early_stopping.early_stop:
                        convergence_epoch = epoch
                        break

            self.total_time = time.time() - total_start_time
            time_to_convergence = sum(m.epoch_duration for m in [train_metrics] if hasattr(m,'epoch_duration')
                                      ) if convergence_epoch >=0 else self.total_time
            avg_epoch_time = sum(epoch_durations) / len(epoch_durations) if epoch_durations else 0
            print("Training abgeschlossen. Speichere Ergebnisse...")
            self.saver.write_run_yaml_summary(config=self.config,
                                              total_training_time=self.total_time,
                                              train_metrics=best_train_metrics,
                                              test_metrics=best_test_metrics,
                                              early_stop_info=self.early_stopping.get_stop_info(),
                                              avg_train_epoch_time= avg_epoch_time,
                                              time_to_convergence=time_to_convergence
                                              )

            self.saver.save_model_and_config_after_epochs(config=self.config,
                                                          model=self.trained_model,
                                                          save_full_model=True
                                                          )

            self.saver.write_multi_run_results_csv(config=self.config,
                                                   total_training_time=self.total_time,
                                                   train_metrics=best_train_metrics,
                                                   test_metrics=best_test_metrics,
                                                   early_stop_info=self.early_stopping.get_stop_info() if self.early_stopping else None,
                                                   run_number=self.run_number,
                                                   avg_train_epoch_time= avg_epoch_time,
                                                   time_to_convergence=time_to_convergence
                                                   )



        except Exception as e:
            # Fehlerbehandlung
            raise e
