import torch
from torch import nn
from helpers.trainer_class import BaseTrainer

class ForwardGradientTrainer(BaseTrainer):
    """Trainer-Klasse für Backpropagation-basiertes Training."""

    def _train_epoch_impl(self):
        sum_loss = 0
        sum_correct = 0
        sum_size = 0

        pbar = self._create_progress_bar(desc=f'BP - Train: {self.epoch_num}/{self.total_epochs}')

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # Forward Pass
            named_params = dict(self.model.named_parameters())
            params = tuple(named_params.values())
            names = tuple(named_params.keys())

            # Pertubation Vektor
            v_params = tuple(torch.randn_like(p) for p in params)
            outputs = self.model(inputs)


            # Define loss function for functional call
            def loss_fn(params_tuple, inputs, targets):
                # Reconstruct parameter dict from tuple
                params_dict = dict(zip(names, params_tuple))
                output = torch.func.functional_call(self.model, params_dict, inputs)
                return nn.functional.cross_entropy(output, targets)

            # JVP
            loss , dir_der = torch.func.jvp(lambda params: loss_fn(params, inputs, targets),
                                               (params,),
                                               (v_params,))
            estimated_gradients = []
            # set Gradients = v*jvp (scalar multiplication)

            with torch.no_grad():
                for j, param in enumerate(self.model.parameters()):
                    estimated_gradient = dir_der * v_params[j]
                    param.grad = estimated_gradient
                    estimated_gradients.append(estimated_gradient)

            sum_loss += loss.item()

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
        pass