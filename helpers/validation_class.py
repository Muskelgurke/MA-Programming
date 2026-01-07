import torch
import helpers.datasets as datasets_helper
import helpers.loss_function as loss_function_helper

from helpers.saver_class import TorchModelSaver
from helpers.config_class import Config
from helpers.validation_metrics_class import TestMetrics
from tqdm import tqdm


class Tester():

    def __init__(self,config_file: Config, device: torch.device, saver_class: TorchModelSaver, test_loader: torch.utils.data.DataLoader):

        self.config = config_file
        self.device = device
        self.saver = saver_class
        self.test_loader = test_loader

        self._initialize_components()

    def _initialize_components(self):
        self.model = None

        self.loss_function = loss_function_helper.get_loss_function(config=self.config)

        self.seed = self.config.random_seed
        self.epoch_num = 0
        self.total_epochs = self.config.epoch_total
        self.metrics = TestMetrics()

    def validate_epoch(self, epoch_num: int, model: torch.nn.Module) -> TestMetrics:
        self.epoch_num = epoch_num + 1
        self.model = model
        self.metrics.clear_test_metrics()
        self._validate()

        return self.metrics

    def _validate(self):
        sum_loss = 0
        sum_correct = 0
        sum_size = 0

        self.model.eval()

        pbar = self._create_progress_bar(desc=f'Validating: {self.epoch_num}/{self.total_epochs}')

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            valid_loss = self.loss_function(outputs,targets)
            sum_loss += valid_loss.item()

            _ , predicted = torch.max(outputs.data,1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()

            sum_correct += correct
            sum_size += total
            accuracy = 100.0 * sum_correct / sum_size

            pbar.set_postfix({
                'Val Loss': f'{valid_loss:.4f}',
                'Val Acc': f'{accuracy:.2f}%'
            })

        self.metrics.loss_per_epoch = sum_loss / sum_size
        self.metrics.acc_per_epoch = 100.0 * sum_correct / sum_size

    def _create_progress_bar(self, desc: str = None) -> tqdm:
        """Create standardized progress bar."""
        if desc is None:
            desc = f'Epoch {self.epoch_num}/{self.total_epochs}'
        return tqdm(self.test_loader, desc=desc)




