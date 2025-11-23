import torch
import time
import datasets as datsets_helper
import optimizer as optimizer_helper
import loss_function as loss_function_helper
import model as model_helper

from _new.helpers.config_class import Config
from _new.helpers.early_stopping_class import EarlyStopping
from _new.helpers.saver_class import TorchModelSaver
from _new.helpers.training_metrics_class import TrainingMetrics
from abc import ABC, abstractmethod
from tqdm import tqdm

class BaseTrainer(ABC):

    def __init__(self, config_file= Config,
                 device= torch.device,
                 saver_class = TorchModelSaver):

        self.config = config_file
        self.device = device
        self.saver = saver_class

        self._initialize_components()

    def _initialize_components(self):
        self.model = model_helper.get_model(config=self.config).to(self.device)
        self.train_loader, xx = datsets_helper.get_dataloaders(config=self.config,
                                                               device=self.device)
        self.loss_function = loss_function_helper.get_loss_function(config=self.config)
        self.optimizer = optimizer_helper.get_optimizer(config=self.config,
                                                        model=self.model)

        self.metrics = TrainingMetrics()
        self.total_epochs = self.config.epoch_total
        self.epoch_num = 0
        self.seed = self.config.random_seed


    def train_epoch(self, epoch_num: int) -> TrainingMetrics:
        """"Trainiert das Modell für eine Epoche und gibt die Trainingsmetriken zurück."""
        self.epoch_num = epoch_num + 1
        self.model.train()
        self.metrics.clear_batch_metrics()

        self._train_epoch_impl()

        return self.metrics

    @abstractmethod
    def _train_epoch_impl(self):

        pass

    def _update_metrics(self, loss: float, outputs: torch.Tensor, targets: torch.Tensor):
        """Update training metrics."""
        _, predicted = torch.max(outputs.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        accuracy = 100.0 * correct / total

        return loss, correct, total, accuracy

    def _create_progress_bar(self, desc: str = None) -> tqdm:
        """Create standardized progress bar."""
        if desc is None:
            desc = f'Epoch {self.epoch_num}/{self.total_epochs}'
        return tqdm(self.train_loader, desc=desc)









