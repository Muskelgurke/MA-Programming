import torch
import time
import helpers.datasets as datasets_helper
import helpers.model as model_helper
import helpers.loss_function as loss_function_helper
import helpers.optimizer as optimizer_helper
from helpers.config_class import Config
from helpers.saver_class import TorchModelSaver
from helpers.training_metrics_class import TrainMetrics
from abc import ABC, abstractmethod
from tqdm import tqdm
from pathlib import Path

class BaseTrainer(ABC):

    def __init__(self, config_file=Config, device= torch.device, saver_class = TorchModelSaver):

        self.config = config_file
        self.device = device
        self.saver = saver_class
        self.runsPath = str(self.saver.run_dir)
        self._initialize_components()

    def _initialize_components(self):


        self.train_loader, xx = datasets_helper.get_dataloaders(config=self.config,
                                                               device=self.device)
        if len(self.train_loader) == 0:
            raise ValueError(
                f"Train DataLoader ist leer! "
                f"Dataset-Größe < Batch-Size ({self.config.batch_size})"
            )

        sample_batch = next(iter(self.train_loader))

        self.model = model_helper.get_model(config=self.config,sample_batch= sample_batch).to(self.device)
        self.loss_function = loss_function_helper.get_loss_function(config=self.config)
        self.optimizer = optimizer_helper.get_optimizer(config=self.config,
                                                        model=self.model)

        self.metrics = TrainMetrics()
        self.total_epochs = self.config.epoch_total
        self.epoch_num = 0
        self.seed = self.config.random_seed


    def train_epoch(self, epoch_num: int) -> tuple[TrainMetrics, torch.nn.Module]:
        """"Trainiert das Modell für eine Epoche und gibt die Trainingsmetriken zurück."""
        self.epoch_num = epoch_num + 1
        self.model.train()

        epoch_start_time = time.time()

        self._train_epoch_impl()
        epoch_duration = time.time() - epoch_start_time
        self.metrics.epoch_duration = epoch_duration

        return self.metrics, self.model

    @abstractmethod
    def _train_epoch_impl(self):
        pass


    def _create_progress_bar(self, desc: str = None) -> tqdm:
        """Create standardized progress bar."""
        if desc is None:
            desc = f'Epoch {self.epoch_num}/{self.total_epochs}'
        return tqdm(self.train_loader, desc=desc)









