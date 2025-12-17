import torch
import datetime
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

        self.should_track_memory_this_epoch = False
        self.NUM_MEMORY_SNAPSHOTS_IN_EPOCH: int = 3
        self.MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000

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

        self.should_track_memory_this_epoch = epoch_num in self.config.memory_snapshot_epochs

        epoch_start_time = time.time()
        self._train_epoch_impl()
        epoch_duration = time.time() - epoch_start_time
        self.metrics.epoch_duration = epoch_duration


        return self.metrics, self.model

    @abstractmethod
    def _train_epoch_impl(self):
        pass

    def start_record_memory_history(self) -> None:
        torch.cuda.memory._record_memory_history(max_entries=self.MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)

    def stop_record_memory_history(self) -> None:
        torch.cuda.memory._record_memory_history(enabled=None)

    def export_memory_snapshot(self) -> None:
        # Prefix for file names.
        time_stamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        file_name = f"mem_snap_ep_{self.epoch_num}_{time_stamp}"
        file_path = f"{self.runsPath}/{file_name}"

        try:
            print(f"Saving snapshot to local file: {file_path}.pickl")
            torch.cuda.memory._dump_snapshot(f"{file_path}.pickle")
        except Exception as e:
            print(f"Failed to capture memory snapshot {e}")
            return


    def _create_progress_bar(self, desc: str = None) -> tqdm:
        """Create standardized progress bar."""
        if desc is None:
            desc = f'Epoch {self.epoch_num}/{self.total_epochs}'
        return tqdm(self.train_loader, desc=desc)









