import torch
import datasets as datesets_helper
import model as model_helper
import loss_function as loss_function_helper
import optimizer as optimizer_helper
import tqdm

from _new.helpers.tester_metrics_class import TesterMetrics
from _new.helpers.saver_class import TorchModelSaver
from _new.helpers.config_class import Config


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
        self.metrics = TesterMetrics()

    def validate_epoch(self, epoch_num: int) -> TesterMetrics:
        self.epoch_num = epoch_num + 1
        self.metrics.clear_test_metrics()

        self._validate()

        return self.metrics

    def _validate(self):
        self.model.eval()
        pbar = tqdm(self.test_loader, desc=f'Validating{self.epoch_num}/{self.total_epochs}')
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
                batch_size = targets.size(0)

                sum_correct += correct
                sum_size += batch_size

                pbar.set_postfix({
                    'Val Loss': f'{sum_loss / (batch_idx + 1):.4f}',
                    'Val Acc': f'{100.0 * sum_correct / sum_size:.2f}%'
                })
            self.metrics.loss_per_epoch = sum_loss / sum_size
            self.metrics.acc_per_epoch = 100.0 * sum_correct / sum_size





