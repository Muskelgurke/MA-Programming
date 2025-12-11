import torch
from torch import nn
from helpers.trainer_class import BaseTrainer

class ForwardGradientTrainer(BaseTrainer):
    """Trainer-Klasse für Backpropagation-basiertes Training."""

    def _train_epoch_impl(self):
        sum_loss = 0
        sum_correct = 0
        sum_size = 0

        pbar = self._create_progress_bar(desc=f'FGD- Train: {self.epoch_num}/{self.total_epochs}')
        named_params = dict(self.model.named_parameters())
        names = tuple(named_params.keys())
        named_buffers = dict(self.model.named_buffers())

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            self.model.train()
            # Warmup forward pass to ensure buffers are initialized
            with torch.no_grad():
                self.model(inputs)
            self.model.eval()

            buffers = {k: v.to(self.device) for k, v in named_buffers.items()}

            params = tuple(named_params.values())

            # Pertubation Vektor
            v_params = tuple(torch.randn_like(p) for p in params)

            # Define loss function for functional call
            def loss_fn(params_tuple, inputs, targets):
                # Reconstruct parameter dict from tuple
                params_dict = dict(zip(names, params_tuple))
                state_dict = {**params_dict, **buffers} # vorher nur params_dict jetzt mit buffers
                output = torch.func.functional_call(self.model, state_dict, inputs)
                loss = nn.functional.cross_entropy(output, targets)
                return loss, output

            # Forward Pass mit JVP
            loss, dir_der, outputs = torch.func.jvp(lambda params: loss_fn(params, inputs, targets),
                                                     (params,),
                                                     (v_params,),
                                                     has_aux=True)
            sum_loss += loss.item()

            # set Gradients = v*jvp (scalar multiplication)
            with torch.no_grad():
                for j, param in enumerate(self.model.parameters()):
                    estimated_gradient = dir_der * v_params[j]
                    param.grad = estimated_gradient



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