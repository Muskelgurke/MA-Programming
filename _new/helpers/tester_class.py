import torch
import _new.helpers.datasets as datesets_helper
import helpers.model as model_helper
import helpers.loss_function as loss_function_helper
import helpers.optimizer as optimizer_helper

from tqdm import tqdm
from helpers.tester_metrics_class import TestMetrics
from helpers.saver_class import TorchModelSaver
from helpers.config_class import Config


class Tester():

    def __init__(self,
                 config_file: Config,
                 device: torch.device,
                 saver_class: TorchModelSaver):

        self.config = config_file
        self.device = device
        self.saver = saver_class

        self._initialize_components()

    def _initialize_components(self):
        self.model = model_helper.get_model(config=self.config).to(self.device)
        xx , self.test_loader = datesets_helper.get_dataloaders(config=self.config,
                                                                device=self.device)
        self.loss_function = loss_function_helper.get_loss_function(config=self.config)
        self.optimizer = optimizer_helper.get_optimizer(config=self.config,
                                                        model=self.model)
        self.seed = self.config.random_seed
        self.epoch_num = 0
        self.total_epochs = self.config.epoch_total
        self.metrics = TestMetrics()

    def validate_epoch(self, epoch_num: int) -> TestMetrics:
        self.epoch_num = epoch_num + 1
        self.metrics.clear_test_metrics()

        self._validate()

        return self.metrics

    def _validate(self):
        self.model.eval()
        pbar = self._create_progress_bar(desc=f'Validating{self.epoch_num}/{self.total_epochs}')

        with torch.no_grad():
            sum_loss = 0
            sum_correct = 0
            sum_size = 0
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_function(outputs,targets)
                sum_loss += loss.item()
                _ , predicted = torch.max(outputs.data,1)
                correct = (predicted == targets).sum().item()

                sum_correct += correct

                pbar.set_postfix({
                    'Val Loss': f'{sum_loss / (batch_idx + 1):.4f}',
                    'Val Acc': f'{100.0 * sum_correct / sum_size:.2f}%'
                })

            self.metrics.loss_per_epoch = sum_loss / len(self.test_loader)
            self.metrics.acc_per_epoch = 100.0 * sum_correct / len(self.test_loader)

    def _create_progress_bar(self, desc: str = None) -> tqdm:
        """Create standardized progress bar."""
        if desc is None:
            desc = f'Epoch {self.epoch_num}/{self.total_epochs}'
        return tqdm(self.test_loader, desc=desc)




