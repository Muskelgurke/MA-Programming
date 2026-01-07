import torch
import datetime
import time
import helpers.datasets as datasets_helper
import helpers.model as model_helper
import helpers.loss_function as loss_function_helper
import helpers.optimizer as optimizer_helper
import gc
import numpy as np
from helpers.config_class import Config
from helpers.saver_class import TorchModelSaver
from helpers.training_metrics_class import TrainMetrics
from abc import ABC, abstractmethod
from tqdm import tqdm
from pathlib import Path

class BaseTrainer(ABC):

    def __init__(self, config_file=Config, device= torch.device, saver_class = TorchModelSaver, train_loader= torch.utils.data.DataLoader ):

        self.config = config_file
        self.device = device
        self.saver = saver_class
        self.runsPath = str(self.saver.run_dir)
        self.train_loader = train_loader
        self._initialize_components()

        self.should_track_memory_this_epoch = False
        self.NUM_MEMORY_SNAPSHOTS_IN_EPOCH: int = 3
        self.MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000

        self.base_memory_bytes = 0
        self.mem_parameters_bytes = 0
        self.max_memory_bytes = 0

    def _initialize_components(self):

        if len(self.train_loader) == 0:
            raise ValueError(
                f"Train DataLoader ist leer! "
                f"Dataset-Größe < Batch-Size ({self.config.batch_size})"
            )

        sample_batch = next(iter(self.train_loader))
        inputs, targets = sample_batch

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        self.base_memory_bytes = self.get_mem_info()
        print(f"TRAINERCLASS-> DEBUG MEM: Base Memory: {self.base_memory_bytes} bytes")
        self.model = model_helper.get_model(config=self.config,sample_batch= sample_batch).to(self.device)
        self.mem_parameters_bytes = self.get_mem_info() - self.base_memory_bytes
        print(f"TRAINERCLASS-> DEBUG MEM: Post-Model Memory: {self.mem_parameters_bytes} bytes")

        self.loss_function = loss_function_helper.get_loss_function(config=self.config)
        self.optimizer = optimizer_helper.get_optimizer(config=self.config,
                                                        model=self.model)

        self.metrics = TrainMetrics()
        self.metrics.mem_base_bytes = self.base_memory_bytes
        self.metrics.mem_parameter_bytes = self.mem_parameters_bytes

        self.total_epochs = self.config.epoch_total
        self.epoch_num = 0
        self.seed = self.config.random_seed

    def train_epoch(self, epoch_num: int) -> tuple[TrainMetrics, torch.nn.Module]:
        """"Trainiert das Modell für eine Epoche und gibt die Trainingsmetriken zurück."""
        self.epoch_num = epoch_num + 1
        self.model.train()

        self.should_track_memory_this_epoch = self.epoch_num in self.config.memory_snapshot_epochs
        #print(f"Epoch={self.epoch_num}: Memory Tracking = {self.should_track_memory_this_epoch}")
        epoch_start_time = time.time()

        torch.cuda.reset_peak_memory_stats()

        self._train_epoch_impl()

        self.max_memory_bytes = torch.cuda.memory.max_memory_allocated()

        epoch_duration = time.time() - epoch_start_time
        self.metrics.epoch_duration = epoch_duration
        self.metrics.max_mem_bytes = self.max_memory_bytes

        return self.metrics, self.model

    @abstractmethod
    def _train_epoch_impl(self):
        pass

    def start_record_memory_history(self) -> None:
        torch.cuda.memory._record_memory_history(max_entries=self.MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)

    def stop_record_memory_history(self) -> None:
        torch.cuda.memory._record_memory_history(enabled=None)

    def export_memory_snapshot(self, batch_idx: int=None )  -> None:
        # Prefix for file names.
        time_stamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        if batch_idx is not None:
            file_name = f"mem_snap_ep{self.epoch_num}_batch{batch_idx}_{time_stamp}"
        else:
            file_name = f"mem_snap_ep{self.epoch_num}_{time_stamp}"
        file_path = f"{self.runsPath}/{file_name}"

        try:
            print(f"TRAINERCLASS-> Saving snapshot to local file: {file_path}.pickl")
            torch.cuda.memory._dump_snapshot(f"{file_path}.pickle")
        except Exception as e:
            print(f"TRAINERCLASS-> Failed to capture memory snapshot {e}")
            return

    def _create_progress_bar(self, desc: str = None) -> tqdm:
        """Create standardized progress bar."""
        if desc is None:
            desc = f'Epoch {self.epoch_num}/{self.total_epochs}'
        return tqdm(self.train_loader, desc=desc)

    def get_mem_info(self):
        # Zwinge Garbage Collection, um Artefakte zu vermeiden
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated(device=self.device)
        # allocate ist belegter Speicher in Bytes
        mem = torch.cuda.memory_allocated(device=self.device)
        return mem









