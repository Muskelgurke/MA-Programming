import csv
import datetime
import pickle
import json
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Union

import numpy as np

from NNTrainingTorch.helpers import plotting
from NNTrainingTorch.helpers.results_of_epochs import (results_of_epochs)
from NNTrainingTorch.helpers.config_class import Config
from NNTrainingTorch.helpers.Training_Metriken import TrainingMetrics


class TorchModelSaver:
    """Class for saving and loading PyTorch model training sessions"""

    def __init__(self, base_dir: str):
        self.run_dir = Path(base_dir)

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

    def save_plotting_csv(self, training_metrics: TrainingMetrics, test_loss: float = None, test_accuracy: float = None,
                          epoch: int = 0) -> None:
        """Save metrics in a plotting-friendly CSV format with one row per epoch"""
        csv_file_path = self.run_dir / "plotting_metrics.csv"
        file_exists = csv_file_path.exists()

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
            'avg_std_true',
            'num_batches'
        ]

        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only once
            if not file_exists:
                writer.writeheader()

            # Calculate averages from batch metrics
            metrics_dict = training_metrics.to_dict()

            avg_mse = np.mean(training_metrics.mse_true_esti_grads_batch)
            avg_mae = np.mean(training_metrics.mae_true_esti_grad_batch)
            avg_std_diff = np.mean(training_metrics.std_of_difference_true_esti_grads_batch)
            avg_std_esti = np.mean(training_metrics.std_of_esti_grads_batch)
            avg_std_true = np.mean(training_metrics.std_of_true_grads_batch)

            # Write one row per epoch with all relevant metrics
            row_data = {
                'epoch': epoch + 1,
                'avg_train_loss': training_metrics.epoch_avg_train_loss,
                'train_accuracy': training_metrics.train_acc_of_epoch,
                'avg_test_loss': test_loss if test_loss is not None else 'csv_save=None',
                'test_accuracy': test_accuracy if test_accuracy is not None else 'csv_save=None',
                'avg_cosine_similarity': training_metrics.avg_cosine_similarity_of_epoch,
                'avg_mse_grads': avg_mse,
                'avg_mae_grads': avg_mae,
                'avg_std_difference': avg_std_diff,
                'avg_std_estimated': avg_std_esti,
                'avg_std_true': avg_std_true,
                'num_batches': len(metrics_dict['cosine_of_esti_true_grads_batch'])
            }

            writer.writerow(row_data)

    def save_training_metrics_to_csv(self, training_metrics: TrainingMetrics, config: Config, epoch: int) -> None:
        """Save training metrics to CSV file, with config written only once at the top"""
        csv_file_path = self.run_dir / "training_metrics.csv"
        file_exists = csv_file_path.exists()

        # Define all possible fieldnames
        fieldnames = [
            'epoch',
            'metric_type',  # 'epoch_start', 'batch', 'epoch_summary', 'test_summary'
            'batch_id',
            'train_loss',
            'mse_loss',
            'mae_loss',
            'std_difference_true_esti',
            'std_estimated_grads',
            'std_true_grads',
            'cosine_similarity_true_esti',
            'avg_train_loss',
            'train_accuracy',
            'avg_test_loss',
            'test_accuracy',
            'avg_cosine_similarity'
        ]

        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header and config only once
            if not file_exists:
                # Write config as comments
                csvfile.write("# Configuration Parameters:\n")
                config_dict = config.to_dict()
                for key, value in config_dict.items():
                    csvfile.write(f"# {key}: {value}\n")
                csvfile.write("# \n")
                writer.writeheader()

            # Write epoch start marker
            writer.writerow({
                'epoch': epoch + 1,
                'metric_type': 'epoch_start',
                'batch_id': '',
                'train_loss': '',
                'mse_loss': '',
                'mae_loss': '',
                'std_difference_true_esti': '',
                'std_estimated_grads': '',
                'std_true_grads': '',
                'cosine_similarity_true_esti': '',
                'avg_train_loss': '',
                'train_accuracy': '',
                'avg_test_loss': '',
                'test_accuracy': '',
                'avg_cosine_similarity': ''
            })

            # Write batch metrics
            batch_metrics = training_metrics.to_dict()
            max_batches = len(batch_metrics.get('cosine_of_esti_true_grads_batch', []))

            for batch_id in range(max_batches):
                batch_row = {
                    'epoch': epoch + 1,
                    'metric_type': 'batch',
                    'batch_id': batch_id + 1,
                    'train_loss': '',  # Individual batch loss not stored in metrics
                    'mse_loss': batch_metrics['mse_true_esti_grads_batch'][batch_id] if batch_id < len(
                        batch_metrics.get('mse_true_esti_grads_batch', [])) else '',
                    'mae_loss': batch_metrics['mae_true_esti_grad_batch'][batch_id] if batch_id < len(
                        batch_metrics.get('mae_true_esti_grad_batch', [])) else '',
                    'std_difference_true_esti': batch_metrics['std_of_difference_true_esti_grads_batch'][
                        batch_id] if batch_id < len(
                        batch_metrics.get('std_of_difference_true_esti_grads_batch', [])) else '',
                    'std_estimated_grads': batch_metrics['std_of_esti_grads_batch'][batch_id] if batch_id < len(
                        batch_metrics.get('std_of_esti_grads_batch', [])) else '',
                    'std_true_grads': batch_metrics['std_of_true_grads_batch'][batch_id] if batch_id < len(
                        batch_metrics.get('std_of_true_grads_batch', [])) else '',
                    'cosine_similarity_true_esti': batch_metrics['cosine_of_esti_true_grads_batch'][
                        batch_id] if batch_id < len(batch_metrics.get('cosine_of_esti_true_grads_batch', [])) else '',
                    'avg_train_loss': '',
                    'train_accuracy': '',
                    'avg_test_loss': '',
                    'test_accuracy': '',
                    'avg_cosine_similarity': ''
                }
                writer.writerow(batch_row)


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