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

class Trainer:

    def __init(self, config_file= Config, device: torch.device, saver_class = TorchModelSaver):
        self.config = config_file
        self.device = device
        self.start_time = time.time()
        self.saver = saver_class
        # Initialisierung von Laden, Modell, Optimierer, Logger, Saver, Early Stopping
        self._initialize()

    def _initialize(self):
        self.model = model_helper.get_model(config=self.config).to(self.device)
        self.train_loader, xx = datsets_helper.get_dataloaders(config=self.config,device=self.device)
        self.loss_function = loss_function_helper.get_loss_function(config=self.config)
        self.optimizer = optimizer_helper.get_optimizer(config=self.config,model=self.model)
        # Setup Early Stopping
        self.early_stopping = EarlyStopping(patience=self.config.early_stopping_patience,
                                            delta=self.config.early_stopping_delta
                                            ) if self.config.early_stopping else None
        self.metrics = TrainingMetrics()
        self.total_epochs = self.config.epoch_num
        self.epoch_num = 0
        self.seed = self.config.random_seed


    def train_epoch(self, epoch_num: int) -> dict:
        self.epoch_num = epoch_num + 1
        self.model.train()






