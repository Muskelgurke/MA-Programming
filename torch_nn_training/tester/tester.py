import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_nn_training.configuration.config_class import Config
from torch_nn_training.tester.tester_metrics_class import TesterMetrics

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
        self.epoch_num = epoch_num
        # print("Evaluation of Epoch on Test Dataset")
        self.model.eval()
        # Reset metrics
        self.avg_validation_acc_of_epoch = 0
        self.avg_validation_loss = 0
        self.val_loss = 0
        self.test_acc = 0
        self.accumulated_running_loss_over_all_batches = 0
        self.acc_of_all_batches = []

        # F1-Score
        self.y_pred = []
        self.y_true = []
        self.f1_score = 0
        self.precision = 0
        self.recall = 0

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
            for batch_idx, (inputs_from_test_loader, targets_from_test_loader) in enumerate(self.test_loader):

                inputs_from_test_loader = inputs_from_test_loader.to(self.device)
                targets_from_test_loader = targets_from_test_loader.to(self.device)
                outputs = self.model(inputs_from_test_loader)
                validation_loss = self.loss_function(outputs, targets_from_test_loader)

                self.accumulated_running_loss_over_all_batches += validation_loss.item()
                n_correct_samples = 0
                amount_samples = 0

                with torch.no_grad():

                    _, predicted = torch.max(outputs.data, 1)
                    amount_samples += targets_from_test_loader.size(0)
                    n_correct_samples += (predicted == targets_from_test_loader).sum().item()
                    acc_of_batch = 100. * n_correct_samples / amount_samples
                    self.y_pred.extend(predicted.cpu().numpy())
                    self.y_true.extend(targets_from_test_loader.cpu().numpy())
                    sum_correct_samples += n_correct_samples
                    sum_total_samples += amount_samples

                self.acc_of_all_batches.append(acc_of_batch)

                # Tensorboard Logging
                unique_increasing_counter = (self.epoch_num - 1) * len(self.test_loader) + batch_idx
                if self.writer is not None:
                    self.writer.add_scalar('Train/Loss - Batch', validation_loss.item(), unique_increasing_counter)
                    self.writer.add_scalar('Train/Accuracy - Batch', acc_of_batch, unique_increasing_counter)

                pbar.update(1)
                pbar.set_postfix({
                    'Loss': f'{validation_loss.item():.4f}'
                })

            pbar.close()
        # calculating Acc and Loss
        self.metrics.test_acc_per_epoch = sum_correct_samples / sum_total_samples * 100

        self.metrics.test_loss_per_epoch = self.accumulated_running_loss_over_all_batches / len(self.test_loader)

    def _calculate_f1(self):
        # Calculate F1-Score, Precision, Recall
        y_true_tensor = torch.tensor(self.y_true)
        y_pred_tensor = torch.tensor(self.y_pred)
        TP = ((y_pred_tensor == 1) & (y_true_tensor == 1)).sum().item()
        FP = ((y_pred_tensor == 1) & (y_true_tensor == 0)).sum().item()
        FN = ((y_pred_tensor == 0) & (y_true_tensor == 1)).sum().item()

        if TP + FP > 0:
            self.precision = TP / (TP + FP)
        else:
            self.precision = 0

        if TP + FN > 0:
            self.recall = TP / (TP + FN)
        else:
            self.recall = 0

        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0

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



