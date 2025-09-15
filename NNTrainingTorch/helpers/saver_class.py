import datetime
import pickle
import json
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Union

import numpy as np

from NNTrainingTorch.helpers import plotting
from NNTrainingTorch.helpers.TrainingResults import (TrainingResults)
from NNTrainingTorch.helpers.config_class import Config


class TorchModelSaver:
    """Class for saving and loading PyTorch model training sessions"""

    def __init__(self, base_dir: str = "../NNTrainingTorch/training_runs"):
        self.base_dir = Path(base_dir)

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

    def save_training_session(self, training_result: TrainingResults,
                              config: Config, model: torch.nn.Module,
                              save_full_model: bool = False) -> Path:
        """
        Save complete PyTorch training session including model, metrics, and configuration.

        Args:
            training_result: TrainingResults Dataclass containing metrics and times
            config: Training configuration dictionary
            model: Trained PyTorch model
            save_full_model: If True, saves full model; otherwise saves state_dict

        Returns:
            Path to the created training directory
        """
        train_accs = training_result.train_accs
        test_accs = training_result.test_accs
        train_losses = training_result.train_losses
        test_losses = training_result.test_losses
        epoch_times = training_result.epoch_times

        # Create timestamp-based directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        training_dir = self.base_dir / timestamp
        training_dir.mkdir(parents=True, exist_ok=True)

        # Prepare training data for TOML
        training_data = config.to_dict()



        yaml_path = training_dir / "training_info.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(training_data, f)

        # Save PyTorch model
        model_dir = training_dir / "model"
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
            with open(model_dir / "model_info.json", 'w') as f:
                json.dump(model_info, f, indent=2)

        # Save optimizer state if available in training_result
        if hasattr(training_result, 'optimizer_state') and training_result.optimizer_state:
            torch.save(training_result.optimizer_state, model_dir / "optimizer_state.pth")

        # Generate plots
        plotting.plot_performance(train_losses, test_losses,train_accs, test_accs,
                                  epoch_times, training_dir, config.dataset_name)

        # Save configuration as JSON
        with open(training_dir / "config.json", 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        # Save detailed training log
        self._save_training_log(training_dir, train_accs, test_accs,
                                train_losses, test_losses, epoch_times)

        return training_dir

    def _save_training_log(self, training_dir: Path, train_accs: list,
                           test_accs: list, train_losses: list,
                           test_losses: list, epoch_times: list) -> None:
        """Save detailed training log"""
        with open(training_dir / "training_log.txt", 'w') as log_file:
            total_epochs = len(train_accs)
            for epoch in range(len(train_accs)):
                epoch_time = epoch_times[epoch]
                train_acc = train_accs[epoch]
                test_acc = test_accs[epoch]
                train_loss = train_losses[epoch]
                test_loss = test_losses[epoch]
                log_file.write(
                    f"Epoch {epoch + 1:2d}/{total_epochs} | "
                    f"Zeit: {epoch_time:5.2f}s | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Test Acc: {test_acc:.4f} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Test Loss: {test_loss:.4f}\n"
                )
            log_file.write(f"Training abgeschlossen! Durchschnittliche Zeit pro Epoch: {np.mean(epoch_times):.2f}s")

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