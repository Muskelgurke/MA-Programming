import csv
import json
import torch
import yaml
from pathlib import Path
from typing import Dict, Any

from torch_nn_training.saver.results_of_epochs import (results_of_epochs)
from torch_nn_training.configuration.config_class import Config
from torch_nn_training.trainer.training_metrics_class import TrainingMetrics


class TorchModelSaver:
    """Class for saving and loading PyTorch model training sessions"""

    def __init__(self, base_dir: str):
        self.run_dir = Path(base_dir)
        self.batch_metrics_file = self.run_dir / "batch_metrics.csv"
        self.epoch_metrics_file = self.run_dir / "epoch_metrics.csv"
        self.epoch_csv_initialized = False
        self.batch_csv_initialized = False

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

    def save_session(self, results_of_epoch: results_of_epochs,
                     config: Config, model: torch.nn.Module,
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
        train_accs = results_of_epoch.train_accs
        test_accs = results_of_epoch.test_accs
        train_losses = results_of_epoch.train_losses
        test_losses = results_of_epoch.test_losses
        epoch_times = results_of_epoch.epoch_times

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

    def write_epoch_metrics(self,
                            training_metrics: TrainingMetrics,
                            test_loss: float = None,
                            test_accuracy: float = None,
                            epoch: int = 0) -> None:
        """Save metrics in a plotting-friendly CSV format with one row per epoch"""
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
        # Initialize CSV file with headers if not done yet
        if not self.epoch_csv_initialized:
            with open(self.epoch_metrics_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            self.epoch_csv_initialized = True

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





    def write_batch_metrics(self, epoch: int, batch_idx: int, loss: float,
                           accuracy: float = None, **kwargs) -> None:
        """Write batch-level metrics to CSV file during training"""

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

        # Initialize CSV file with headers if not done yet
        if not self.batch_csv_initialized:
            with open(self.batch_metrics_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            self.batch_csv_initialized = True

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


    def load_training_session(self, training_dir: Path,
                              model_class=None, model_kwargs: Dict = None) -> Dict[str, Any]:
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