from _new.helpers.trainer_class import BaseTrainer

class BackpropTrainer(BaseTrainer):
    """Trainer-Klasse für Backpropagation-basiertes Training."""

    def _train_epoch_impl(self):
        sum_loss = 0
        sum_correct = 0
        sum_total = 0

        pbar = self._create_progress_bar(desc=f'BP Epoch {self.epoch_num}/{self.total_epochs}')

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # Metriken aktualisieren
            loss_val, correct, total, train_acc = self._update_metrics(loss.item(), outputs, targets)
            sum_loss += loss_val
            sum_correct += correct
            sum_total += total

            pbar.set_postfix({
                'Loss': f'{sum_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100.0 * sum_correct / sum_total:.2f}%'
            })

        # Epoch-Metriken speichern
        self.metrics.acc_per_epoch = 100. * sum_correct / sum_total
        self.metrics.loss_per_epoch = sum_loss / len(self.train_loader)
        self.metrics.num_batches = len(self.train_loader)