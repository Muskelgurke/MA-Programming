import csv
import json
import torch
import yaml
import datetime
from pathlib import Path
from typing import Dict, Any, TextIO, Optional

from configuration.config_class import Config
from trainer.training_metrics_class import TrainingMetrics


class TorchModelSaver:
    """Class for saving and loading PyTorch model training sessions"""

    def __init__(self, base_dir: str):
        self.run_dir = Path(base_dir)
        self.batch_metrics_file = self.run_dir / "batch_metrics.csv"
        self.epoch_metrics_file = self.run_dir / "epoch_metrics.csv"
        self.epoch_csv_initialized = False
        self.batch_csv_initialized = False

        self._epoch_file_handle: Optional[TextIO] = None
        self._batch_file_handle: Optional[TextIO] = None

    def _ensure_epoch_csv_ready(self):
        """Initialisiert die Epoch CSV-Datei falls nötig"""
        if not self.epoch_csv_initialized:
            fieldnames = [
                'epoch', 'avg_train_loss', 'train_accuracy', 'avg_test_loss',
                'test_accuracy', 'avg_cosine_similarity', 'avg_mse_grads',
                'avg_mae_grads', 'avg_std_difference', 'avg_std_estimated',
                'avg_var_estimated', 'avg_std_true', 'avg_var_true', 'num_batches'
            ]

            with open(self.epoch_metrics_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            self.epoch_csv_initialized = True

    def _ensure_batch_csv_ready(self):
        """Initialisiert die Batch CSV-Datei falls nötig"""
        if not self.batch_csv_initialized:
            fieldnames = [
                'epoch', 'batch_idx', 'loss', 'accuracy', 'cosine_similarity',
                'mse_grads', 'mae_grads', 'std_difference', 'std_estimated',
                'var_estimated', 'std_true', 'var_true'
            ]

            with open(self.batch_metrics_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            self.batch_csv_initialized = True


    def write_epoch_metrics(self,
                            training_metrics: TrainingMetrics,
                            test_loss: float = None,
                            test_accuracy: float = None,
                            epoch: int = 0) -> None:
        """Save metrics in a plotting-friendly CSV format with one row per epoch"""

        self._ensure_epoch_csv_ready()

        # Define fieldnames optimized for plotting
        fieldnames = [
            'epoch',
            'avg_train_loss',
            'train_accuracy',
            'avg_test_loss',
            'test_accuracy',
            'avg_cosine_similarity',
            'avg_mse_grads',
            'avg_mae_grads',
            'avg_std_difference',
            'avg_std_estimated',
            'avg_var_estimated',
            'avg_std_true',
            'avg_var_true',
            'num_batches'
        ]

        # Write one row per epoch with all relevant metrics
        row_data = {
            'epoch': epoch + 1,
            'avg_train_loss': training_metrics.epoch_avg_train_loss,
            'train_accuracy': training_metrics.epoch_train_acc,
            'avg_test_loss': test_loss if test_loss is not None else 'csv_save=None',
            'test_accuracy': test_accuracy if test_accuracy is not None else 'csv_save=None',
            'avg_cosine_similarity': training_metrics.epoch_avg_cosine_similarity,
            'avg_mse_grads': training_metrics.epoch_avg_mse_grads,
            'avg_mae_grads': training_metrics.epoch_avg_mae_grads,
            'avg_std_difference': training_metrics.epoch_avg_std_difference,
            'avg_std_estimated': training_metrics.epoch_avg_std_estimated,
            'avg_std_true': training_metrics.epoch_avg_std_true,
            'avg_var_estimated': training_metrics.epoch_avg_var_estimated,
            'avg_var_true': training_metrics.epoch_avg_var_true,
            'num_batches': training_metrics.num_batches
        }

        with open(self.epoch_metrics_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row_data)
            csvfile.flush()

    def write_batch_metrics(self, epoch: int, batch_idx: int, loss: float,
                            accuracy: float = None, **kwargs) -> None:
        """Write batch-level metrics to CSV file during training"""

        self._ensure_batch_csv_ready()

        # Define fieldnames for batch metrics
        fieldnames = [
            'epoch',
            'batch_idx',
            'loss',
            'accuracy',
            'cosine_similarity',
            'mse_grads',
            'mae_grads',
            'std_difference',
            'std_estimated',
            'var_estimated',
            'std_true',
            'var_true'
        ]

        # Prepare row data
        row_data = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'loss': loss,
            'accuracy': accuracy if accuracy is not None else '',
            'cosine_similarity': kwargs.get('cosine_similarity', ''),
            'mse_grads': kwargs.get('mse_grads', ''),
            'mae_grads': kwargs.get('mae_grads', ''),
            'std_difference': kwargs.get('std_difference', ''),
            'std_estimated': kwargs.get('std_estimated', ''),
            'var_estimated': kwargs.get('var_estimated', ''),
            'std_true': kwargs.get('std_true', ''),
            'var_true': kwargs.get('var_true', '')
        }

        # Write row to CSV
        with open(self.batch_metrics_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row_data)
            csvfile.flush()

    def write_run_summary(self, config: Config,
                          total_training_time: float,
                          train_acc: float = None,
                          test_acc: float = None,
                          test_loss: float = None,
                          early_stop_info: Dict = None) -> None:
        """Save run information including configuration and total training time"""
        # Prepare run summary
        run_info = {
            "total_training_time_seconds": total_training_time
        }

        # Add final metrics if provided
        if train_acc is not None:
            run_info["final_train_accuracy"] = train_acc
        if test_acc is not None:
            run_info["final_test_accuracy"] = test_acc
        if test_loss is not None:
            run_info["final_test_loss"] = test_loss

        # Add early stopping information if provided
        if early_stop_info is not None:
            run_info["early_stopping"] = early_stop_info

        # Save run summary to YAML
        run_info_path = self.run_dir / "run_summary.yaml"
        with open(run_info_path, 'w') as f:
            yaml.dump(run_info, f, default_flow_style=False)

    def cleanup(self):
        """Schließe alle offenen File-Handles"""
        if self._epoch_file_handle:
            self._epoch_file_handle.close()
            self._epoch_file_handle = None
        if self._batch_file_handle:
            self._batch_file_handle.close()
            self._batch_file_handle = None


    def write_multi_session_summary(self,
                                    successful_runs: list,
                                    failed_runs: list) -> None:
        """Save multi-run training summary to CSV with ranking"""
        timestamp = datetime.datetime.now().strftime("%H_%M_%S")
        today = datetime.datetime.now().strftime("%Y_%m_%d")
        day_path = Path(f"runs/{today}/{timestamp}_multi_run_summary")

        day_path.mkdir(parents=True, exist_ok=True)

        # CSV für erfolgreiche Läufe
        if successful_runs:
            # Nach Test-Accuracy sortieren
            successful_runs.sort(key=lambda x: x['final_test_acc'], reverse=True)

            successful_csv_path = day_path / "xx_multi_run_rank_summary.csv"
            fieldnames = [
                'rank',
                'run_id',
                'learning_rate',
                'training_method',
                'random_seed',
                'final_train_acc',
                'final_test_acc',
                'final_train_loss',
                'final_test_loss',
                'total_training_time'
            ]

            with open(successful_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for rank, result in enumerate(successful_runs, 1):
                    config = result['config']
                    row_data = {
                        'rank': rank,
                        'run_id': result['run'],
                        'learning_rate': config['learning_rate'],
                        'training_method': config['training_method'],
                        'random_seed': config['random_seed'],
                        'final_train_acc': result['final_train_acc'],
                        'final_test_acc': result['final_test_acc'],
                        'final_train_loss': result['final_train_loss'],
                        'final_test_loss': result['final_test_loss'],
                        'total_training_time': result.get('total_training_time', '')
                    }
                    writer.writerow(row_data)
                csvfile.flush()

        # CSV für fehlgeschlagene Läufe
        if failed_runs:
            failed_csv_path = day_path / "xx_multi_run_results_failed.csv"
            fieldnames = ['run_id', 'error', 'learning_rate', 'training_method', 'random_seed']

            with open(failed_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in failed_runs:
                    config = result.get('config', {})
                    row_data = {
                        'run_id': result['run'],
                        'error': result['error'],
                        'learning_rate': config.get('learning_rate', ''),
                        'training_method': config.get('training_method', ''),
                        'random_seed': config.get('random_seed', '')
                    }
                    writer.writerow(row_data)
                csvfile.flush()

        # Kombinierte Übersicht
        overview_csv_path = day_path / "xx_multi_run_overview.csv"
        with open(overview_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Summary', 'Count'])
            writer.writerow(['Total Runs', len(successful_runs) + len(failed_runs)])
            writer.writerow(['Successful Runs', len(successful_runs)])
            writer.writerow(['Failed Runs', len(failed_runs)])

            if successful_runs:
                writer.writerow([])
                writer.writerow(['Best Result', ''])
                best = successful_runs[0]
                writer.writerow(['Best Test Accuracy', f"{best['final_test_acc']:.2f}%"])
                writer.writerow(['Best run_id', best['run']])
                writer.writerow(['Best Learning Rate', best['config']['learning_rate']])
                writer.writerow(['Best Training Method', best['config']['training_method']])
                writer.writerow(['Best Random Seed', best['config']['random_seed']])

            csvfile.flush()
        print(f"Multi-run Ergebnisse gespeichert in:")
        print(f"  - {successful_csv_path}")
        if failed_runs:
            print(f"  - {failed_csv_path}")
        print(f"  - {overview_csv_path}")

    def save_torch_model(self, model: torch.nn.Module, filepath: Path,
                         save_state_dict: bool = True) -> None:
        """Save PyTorch model with state dict or full model"""
        if save_state_dict:
            torch.save(model.state_dict(), filepath.with_suffix('.pth'))
        else:
            torch.save(model, filepath.with_suffix('.pt'))

    def load_torch_model(self, filepath: Path, model_class=None,
                         model_kwargs: Dict = None) -> torch.nn.Module:
        """Load PyTorch model from file"""
        if filepath.suffix == '.pth':
            # Loading state dict requires model class
            if model_class is None:
                raise ValueError("model_class required when loading state dict")
            model = model_class(**(model_kwargs or {}))
            model.load_state_dict(torch.load(filepath, map_location='cpu'))
            return model
        else:
            # Loading full model
            return torch.load(filepath, map_location='cpu')

    def save_session(self,config: Config,
                     model: torch.nn.Module,
                     save_full_model: bool = False) -> Path:
        """
        Save complete PyTorch training session including model, metrics, and configuration.

        Args:
            results_of_epoch: TrainingResults Dataclass containing metrics and times
            config: Training configuration dictionary
            model: Trained PyTorch model
            save_full_model: If True, saves full model; otherwise saves state_dict

        Returns:
            Path to the created training directory
        """
       # Prepare training data for yaml
        training_config = config.to_dict()

        yaml_path = self.run_dir / "training_info.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(training_config, f)

        # Save PyTorch model
        model_dir = self.run_dir / "model"
        model_dir.mkdir(exist_ok=True)

        if save_full_model:
            # Save complete model
            torch.save(model, model_dir / "full_model.pt")
        else:
            # Save state dict (recommended)
            torch.save(model.state_dict(), model_dir / "model_state_dict.pth")

            # Save model architecture info
            model_info = {
                "model_class": model.__class__.__name__,
                "model_module": model.__class__.__module__,
                "state_dict_keys": list(model.state_dict().keys())
            }

        return self.run_dir


    def load_training_session(self, training_dir: Path,
                              model_class=None,
                              model_kwargs: Dict = None) -> Dict[str, Any]:
        """Load complete training session"""
        # Load configuration
        with open(training_dir / "config.json", 'r') as f:
            config = json.load(f)

        # Load training info
        with open(training_dir / "training_info.yaml", 'r') as f:
            training_info = yaml.load(f)

        # Load model
        model_dir = training_dir / "model"
        model = None

        if (model_dir / "full_model.pt").exists():
            model = torch.load(model_dir / "full_model.pt", map_location='cpu')
        elif (model_dir / "model_state_dict.pth").exists():
            if model_class is not None:
                model = model_class(**(model_kwargs or {}))
                model.load_state_dict(torch.load(model_dir / "model_state_dict.pth", map_location='cpu'))

        return {
            "model": model,
            "config": config,
            "training_info": training_info,
            "training_dir": training_dir
        }