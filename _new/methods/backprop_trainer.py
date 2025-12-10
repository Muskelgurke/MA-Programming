from _new.helpers.trainer_class import BaseTrainer
from torch.profiler import profile, ProfilerActivity, record_function
import torch



class BackpropTrainer(BaseTrainer):
    """Trainer-Klasse für Backpropagation-basiertes Training."""

    def _train_epoch_impl(self):
        sum_loss = 0
        sum_correct = 0
        sum_size = 0

        pbar = self._create_progress_bar(desc=f'BP - Train: {self.epoch_num}/{self.total_epochs}')

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            sum_loss += loss.item()
            loss.backward()


            self.optimizer.step()

            # Metriken aktualisieren
            _, predicted = torch.max(outputs.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()

            sum_correct += correct
            sum_size += total
            accuracy = 100.0 * sum_correct / sum_size

            pbar.set_postfix({
                'Loss': f'{loss:.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        self.metrics.loss_per_epoch = sum_loss / sum_size
        self.metrics.acc_per_epoch = 100. * sum_correct / sum_size
        self.metrics.num_batches = sum_size