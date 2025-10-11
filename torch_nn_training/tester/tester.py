import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from configuration.config_class import Config
from tester.tester_metrics_class import TesterMetrics

class Tester:
    def __init__(self,
                 config_file: Config,
                 model: torch.nn.Module,
                 test_loader: torch.utils.data.DataLoader,
                 loss_function: torch.nn.Module,
                 device: torch.device,
                 total_epochs: int,
                 seed: int,
                 tensorboard_writer: torch.utils.tensorboard.SummaryWriter):

        self.config = config_file
        self.model = model
        self.test_loader = test_loader
        self.loss_function = loss_function
        self.device = device
        self.total_epochs = total_epochs
        self.random_seed = seed
        self.val_loss = 0
        self.test_acc = 0

        # logging
        self.writer = tensorboard_writer

        # Metrics
        self.avg_validation_acc_of_epoch = 0
        self.avg_validation_loss = 0
        self.sum_correct_samples = 0
        self.accumulated_total_samples_of_all_batches = 0
        self.accumulated_running_loss_over_all_batches = 0
        self.acc_of_all_batches = []
        # F1-Score
        self.f1_score = 0
        self.precision = 0
        self.recall = 0

        self.metrics = TesterMetrics()
        self.epoch_num = 0

    def validate_epoch(self, epoch_num: int)-> TesterMetrics:
        self.epoch_num = epoch_num + 1
        # print("Evaluation of Epoch on Test Dataset")
        self.model.eval()
        # Reset metrics
        self.avg_validation_acc_of_epoch = 0
        self.avg_validation_loss = 0
        self.val_loss = 0
        self.test_acc = 0
        self.accumulated_running_loss_over_all_batches = 0
        self.acc_of_all_batches = []

        if self.config.model_type == "demo_linear_regression":
            self.eval_linearRegression()
        else:
            self.eval_classification()

        return self.metrics

    def eval_classification(self):
        with torch.no_grad():  # Disable gradient computation for efficiency
            pbar = tqdm(self.test_loader, desc=f'Test Epoch {self.epoch_num}/{self.total_epochs}')
            sum_correct_samples = 0
            sum_total_samples = 0
            sum_loss = 0

            all_predictions = []
            all_targets = []

            for batch_idx, (inputs, targets) in enumerate(self.test_loader):

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                validation_loss = self.loss_function(outputs, targets)

                sum_loss += validation_loss.item()

                _, predicted = torch.max(outputs.data, 1)

                batch_correct = (predicted == targets).sum().item()
                batch_size = targets.size(0)

                sum_correct_samples += batch_correct
                sum_total_samples += batch_size

                acc_of_batch = 100. * batch_correct / batch_size

                all_predictions.append(predicted)
                all_targets.append(targets)

                self.acc_of_all_batches.append(acc_of_batch)

                # Tensorboard Logging
                unique_increasing_counter = (self.epoch_num - 1) * len(self.test_loader) + batch_idx
                if self.writer is not None:
                    self.writer.add_scalar('Train/Loss - Batch', validation_loss.item(), unique_increasing_counter)
                    self.writer.add_scalar('Train/Accuracy - Batch', acc_of_batch, unique_increasing_counter)

                pbar.update(1)
                pbar.set_postfix({
                    'Loss': f'{validation_loss.item():.4f}',
                    'ACC': f' {100. * sum_correct_samples / sum_total_samples:.2f}%'
                })
                if torch.cuda.is_available():
                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache()

            pbar.close()

            # calculating Acc and Loss
            self.metrics.test_acc_per_epoch = sum_correct_samples / sum_total_samples * 100

            self.metrics.test_loss_per_epoch = sum_loss / len(self.test_loader)


    def eval_linearRegression(self):
        with torch.no_grad():  # Disable gradient computation for efficiency
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                test_loss = self.loss_function(outputs, targets)
                self.avg_validation_loss += test_loss.item() * data.size(0)
                self.accumulated_total_samples_of_all_batches += data.size(0)

        self.avg_validation_acc_of_epoch = np.mean(self.acc_of_all_batches)
        self.test_acc = self.sum_correct_samples / self.accumulated_total_samples_of_all_batches
        self.avg_validation_loss = self.accumulated_running_loss_over_all_batches / len(self.test_loader)



