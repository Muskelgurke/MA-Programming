import torch
from helpers.config_class import Config
from helpers.saver_class import TorchModelSaver
from helpers.training_metrics_class import TrainMetrics
import helpers.datasets as datasets_helper
import helpers.model as model_helper
import helpers.loss_function as loss_function_helper
import helpers.optimizer as optimizer_helper
from abc import ABC, abstractmethod
from tqdm import tqdm

class BaseTrainer(ABC):

    def __init__(self, config_file=Config, device= torch.device, saver_class = TorchModelSaver):

        self.config = config_file
        self.device = device
        self.saver = saver_class

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

        self._train_epoch_impl()

        return self.metrics, self.model

    @abstractmethod
    def _train_epoch_impl(self):
        pass


    def _create_progress_bar(self, desc: str = None) -> tqdm:
        """Create standardized progress bar."""
        if desc is None:
            desc = f'Epoch {self.epoch_num}/{self.total_epochs}'
        return tqdm(self.train_loader, desc=desc)


    def extract_profiler_metric(self, profiler, key_name, metric_attr="self_cuda_memory_usage"):
        """
        Hilfsfunktion die Profiler-Metriken zu extrahieren für die Validation_metrics_DataClass.

        Torch.profiler wird in _train_epoch_impl() verwendet.

        """
        # key_averages() aggregiert alle Events
        averages = profiler.key_averages()
        for event in averages:
            if event.key == key_name:
                # self_cuda_memory_usage gibt an, was die Funktion selbst allozierte (ohne Kinder)
                # cuda_memory_usage inkludiert Kinder.
                val_bytes = getattr(event, metric_attr)
                return val_bytes / (1024 * 1024)  # Umrechnung in MB
        return 0.0







