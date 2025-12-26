import csv
import torch
import yaml
import os
from pathlib import Path
from helpers.config_class import Config
from helpers.training_metrics_class import TrainMetrics
from helpers.validation_metrics_class import TestMetrics


class TorchModelSaver:
    """Class for saving and loading PyTorch model training sessions"""

    def __init__(self, base_path: str, run_path: str):
        self.base_dir = Path(base_path)
        self.run_dir = Path(run_path)
        self.batch_metrics_file = self.run_dir / "batch_metrics.csv"
        self.epoch_metrics_file = self.run_dir / "epoch_metrics.csv"
        self.summary_file = self.base_dir / "multi_run_summary.csv"
        self.epoch_csv_initialized = False
        self.batch_csv_initialized = False

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_multi_run_csv_ready()

    def _ensure_multi_run_csv_ready(self):
        """Initialisiert die Multi-Run CSV-Datei mit Header falls nötig"""
        if not self.summary_file.exists():
            # Define all fieldnames that will be used
            fieldnames = [
                "run_number",
                "total_training_time_seconds",
                "train_accuracy",
                "train_loss",
                "test_accuracy",
                "test_loss",
                "early_stopped_at_epoch",
                "early_stopping_reason",
                "early:stopping_delta",
                "early:stopping_patience",
                "epoch_total",
                "random_seed",
                "dataset_name",
                "model_type",
                "training_method",
                "learning_rate",
                "optimizer_type",
                "optimizer_momentum",
                "batch_size",
                "time_to_convergence_seconds",
                "avg_train_epoch_time_seconds",
                "best_train_mem_base_bytes",
                "best_train_mem_parameter_bytes",
                "best_train_avg_mem_forward_pass_bytes",
                "best_train_max_mem_forward_pass_bytes",
                "best_train_max_mem_bytes",
            ]

            with open(self.summary_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    def _ensure_epoch_csv_ready(self):
        """Initialisiert die Epoch CSV-Datei falls nötig"""
        if not self.epoch_csv_initialized:
            fieldnames = [
                'epoch',
                'train_loss',
                'train_accuracy',
                'test_loss',
                'test_accuracy',
                'num_batches',
                'early_stop_reason',
                'epoch_duration',
                'time_to_converge',
                'mem_base_bytes',
                'mem_parameter_bytes',
                'avg_mem_forward_pass_bytes',
                'max_mem_forward_pass_bytes',
                'max_mem_bytes'
            ]

            with open(self.epoch_metrics_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            self.epoch_csv_initialized = True

    def _ensure_batch_csv_ready(self):
        """Initialisiert die Batch CSV-Datei falls nötig"""
        if not self.batch_csv_initialized:
            fieldnames = [
                'epoch', 'batch_idx', 'loss', 'accuracy'
            ]

            with open(self.batch_metrics_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            self.batch_csv_initialized = True

    def write_epoch_metrics_csv(self,
                                train_metrics: TrainMetrics,
                                test_metrics: TestMetrics,
                                epoch_idx: int=0,
                                early_stop_reason: dict='') -> None:
        """Save metrics in a plotting-friendly CSV format with one row per epoch.
        Coulmns:
        epoch, train_loss, train_accuracy, test_loss, test_accuracy, num_batches"""

        self._ensure_epoch_csv_ready()

        # Define fieldnames optimized for plotting
        fieldnames = [
            'epoch',
            'train_loss',
            'train_accuracy',
            'test_loss',
            'test_accuracy',
            'num_batches',
            'early_stop_reason',
            'epoch_duration',
            'time_to_converge',
            'mem_base_bytes',
            'mem_parameter_bytes',
            'avg_mem_forward_pass_bytes',
            'max_mem_forward_pass_bytes',
            'max_mem_bytes'
        ]

        # Write one row per epoch with all relevant metrics
        row_data = {
            'epoch': epoch_idx + 1,
            'train_loss': train_metrics.loss_per_epoch,
            'train_accuracy': train_metrics.acc_per_epoch,
            'test_loss': test_metrics.loss_per_epoch if test_metrics.loss_per_epoch is not None else 'csv_save=None',
            'test_accuracy': test_metrics.acc_per_epoch if test_metrics.acc_per_epoch  is not None else 'csv_save=None',
            'num_batches': train_metrics.num_batches,
            'early_stop_reason': early_stop_reason.get('reason', '') if early_stop_reason else '',
            'epoch_duration': train_metrics.epoch_duration,
            'time_to_converge': train_metrics.time_to_converge,
            'mem_base_bytes': train_metrics.mem_base_bytes,
            'mem_parameter_bytes': train_metrics.mem_parameter_bytes,
            'avg_mem_forward_pass_bytes': train_metrics.avg_mem_forward_pass_bytes,
            'max_mem_forward_pass_bytes': train_metrics.max_mem_forward_pass_bytes,
            'max_mem_bytes': train_metrics.max_mem_bytes,
        }

        with open(self.epoch_metrics_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row_data)
            csvfile.flush()

    def write_batch_metrics_csv(self, epoch: int, batch_idx: int, loss: float,
                                accuracy: float = None, **kwargs) -> None:
        """Write batch-level metrics to CSV file during training"""

        self._ensure_batch_csv_ready()

        # Define fieldnames for batch metrics
        fieldnames = [
            'epoch',
            'batch_idx',
            'loss',
            'accuracy'
        ]

        # Prepare row data
        row_data = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'loss': loss,
            'accuracy': accuracy if accuracy is not None else ''
        }

        # Write row to CSV
        with open(self.batch_metrics_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row_data)
            csvfile.flush()

    def write_run_yaml_summary(self, config: Config,
                               total_training_time: float,
                               train_metrics: TrainMetrics,
                               test_metrics: TestMetrics,
                               early_stop_info: dict,
                               avg_train_epoch_time: float,
                               time_to_convergence: float) -> None:

        """Save run information including configuration and total training time"""
        run_summary = {
            "total_training_time_seconds": total_training_time,
            "time_to_convergence_seconds": time_to_convergence,
            "avg_train_epoch_time_seconds": avg_train_epoch_time,
            "train_accuracy": train_metrics.acc_per_epoch,
            "train_loss": train_metrics.loss_per_epoch,
            "test_accuracy": test_metrics.acc_per_epoch,
            "test_loss": test_metrics.loss_per_epoch,
            "early_stopped_at_epoch": early_stop_info.get('stopped_epoch', None),
            "early_stopping_reason": early_stop_info.get('reason', None),
            "configuration": config.to_dict()
        }

        # Save run summary to YAML
        run_summary_path = self.run_dir / "run_summary.yaml"
        with open(run_summary_path, 'w') as f:
            yaml.dump(run_summary, f, default_flow_style=False)

    def write_multi_run_results_csv(self, config: Config,
                                    total_training_time: float,
                                    train_metrics: TrainMetrics,
                                    test_metrics: TestMetrics,
                                    early_stop_info: dict,
                                    run_number: int,
                                    avg_train_epoch_time: float,
                                    time_to_convergence: float) -> None:
        """Save run information including configuration and total training time to CSV"""

        run_summary = {
            "run_number": run_number,
            "total_training_time_seconds": total_training_time,
            "train_accuracy": train_metrics.acc_per_epoch,
            "train_loss": train_metrics.loss_per_epoch,
            "test_accuracy": test_metrics.acc_per_epoch,
            "test_loss": test_metrics.loss_per_epoch,
            "early_stopped_at_epoch": early_stop_info.get('stopped_at_epoch', None) if early_stop_info else None,
            "early_stopping_reason": early_stop_info.get('reason', None) if early_stop_info else None,
            "early:stopping_delta": config.early_stopping_delta,
            "early:stopping_patience": config.early_stopping_patience,
            "epoch_total": config.epoch_total,
            "random_seed": config.random_seed,
            "dataset_name": config.dataset_name,
            "model_type": config.model_type,
            "training_method": config.training_method,
            "learning_rate": config.learning_rate,
            "optimizer_type": config.optimizer,
            "optimizer_momentum": config.momentum,
            "batch_size": config.batch_size,
            "time_to_convergence_seconds": time_to_convergence,
            "avg_train_epoch_time_seconds": avg_train_epoch_time,
            'best_train_mem_base_bytes': train_metrics.mem_base_bytes,
            'best_train_mem_parameter_bytes': train_metrics.mem_parameter_bytes,
            'best_train_avg_mem_forward_pass_bytes': train_metrics.avg_mem_forward_pass_bytes,
            'best_train_max_mem_forward_pass_bytes': train_metrics.max_mem_forward_pass_bytes,
            'best_train_max_mem_bytes': train_metrics.max_mem_bytes,
        }


        with open(self.summary_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=run_summary.keys())

            writer.writerow(run_summary)
            csvfile.flush()
            os.fsync(csvfile.fileno())

    def save_model_and_config_after_epochs(self, config: Config,
                                           model: torch.nn.Module,
                                           save_full_model: bool = False) -> None:
        """
        Save complete model and configuration after training epochs.
        """
        config_dict = config.to_dict()

        yaml_path = self.run_dir / "config.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f)

        # Save PyTorch model
        model_dir = self.run_dir / "model"
        model_dir.mkdir(exist_ok=True)

        if save_full_model:
            # Save complete model
            torch.save(model, model_dir / "full_model.pt")
        else:
            # Save state dict (recommended)
            torch.save(model.state_dict(), model_dir / "model_state_dict.pth")
