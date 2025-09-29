import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

import numpy as np
from absl.testing.parameterized import named_parameters
from fontTools.misc.timeTools import epoch_diff
from tensorflow import switch_case

from torch import nn
from torch.autograd.forward_ad import unpack_dual
from tqdm import tqdm
from torch.func import functional_call

from NNTrainingTorch.helpers.config_class import Config


class Trainer:
    """
    A trainer class that uses functional programming approach with JVP for gradient computation.
    """

    def __init__(self,
                 config: Config,
                 model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 loss_function: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 total_epochs: int,
                 seed: int):
        """Initialize the trainer with all required parameters."""
        self.model = model
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.epoch_num = 0
        self.total_epochs = total_epochs
        self.seed = seed
        self.config = config

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        match self.config.training_method:
            case "fgd":
                print("Using Forward Gradient Descent Method")
            case "bp":
                print("Using Backpropagation Method ")
            case _:
                raise ValueError(f"Unknown Training - Method: {self.config.training_method}")

    def _mse_loss(self, params: dict, x: torch.Tensor, targets: torch.Tensor):
        model = lambda p, x: torch.func.functional_call(self.model, (p, {}), x)
        y = model(params, x)
        return nn.functional.mse_loss(y, targets)

    def _cross_entropy_loss(self, params: dict, x: torch.Tensor, targets: torch.Tensor):
        model = lambda p, x: torch.func.functional_call(self.model, (p, {}), x)
        y = model(params, x)
        return nn.functional.cross_entropy(y, targets)

    def train_epoch(self, epoch_num: int) -> tuple[float, float]:
        """
        Train the model for one epoch using functional forward gradient descent.

        Returns:
            tuple: (average_train_loss, average_train_accuracy)
        """

        self.epoch_num = epoch_num
        self.model.train()

        #ToDo: hier muss noch eine switch case rein damit man zwischen backprop und FGD switchen kann
        # Add deterministic
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

        #ToDo: Backpropagation als alternative Methode einbauen bzw. wechseln je nach Config
        match self.config.training_method:
            case "fgd":
                if self.config.dataset_name == "demo_linear_regression":
                    running_loss, total, correct = self._train_epoch_forward_gradient_linearRegression()
                else:
                    running_loss, total, correct = self._train_epoch_forward_gradient()
            case "bp":
                runnig_loss, total, correct = self._backpropaggation()

            case _:
                raise ValueError(f"Unknown Training - Method: {self.config.training_method}")

        avg_train_loss_of_epoch = running_loss / len(self.data_loader)
        avg_train_acc_of_epoch = 100. * correct / total

        return avg_train_loss_of_epoch, avg_train_acc_of_epoch


    def _train_epoch_forward_gradient(self) -> tuple[float, int, int]:
        """
        Train the model for each epoch using functional forward gradient decent
        :return:
        """
        accumulated_running_loss_over_all_batches = 0.0
        n_correct_samples = 0
        total_amount_of_samples = 0
        pbar = tqdm(self.data_loader, desc=f'FGD Training Epoch {self.epoch_num}/{self.total_epochs}')

        def loss_fn(params_dict):
            return self._cross_entropy_loss(params_dict, inputs, targets)

        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            # initial_state_dict = self.model.state_dict().copy()
            # self.model.load_state_dict(initial_state_dict)
            self.model.train()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            torch.manual_seed(self.seed + batch_idx)
            np.random.seed(self.seed)

            params = dict(self.model.named_parameters())
            loss_value = self._cross_entropy_loss(params, inputs, targets)

            for name, param in self.model.named_parameters():
                #v_param = torch.randn_like(param)
                #ToDo: anschauen weleche anderen Verteilungen es noch so gibt
                v_param=torch.randn(param.shape,
                                    dtype=param.dtype,
                                    device=param.device,
                                    generator=torch.Generator().manual_seed(self.seed + batch_idx + hash(name) % 10000))
                #sv_param = torch.randn(param.shape, dtype=param.dtype, device=param.device, generator=torch.Generator().manual_seed(self.seed+batch_idx))
                #print(f' name: {name}\n v_param:\n {v_param}\n')
                v_params_single = {n: torch.zeros_like(p) if n != name else v_param
                                   for n, p in params.items()}
                #print(f' v_params_single for {name}:\n {v_params_single}\n')
                _,directional_derivatives = torch.func.jvp(
                    loss_fn, (params,), (v_params_single,)
                )
                gradient = directional_derivatives * v_param
                param.grad = gradient

            #print(self.model.parameters())
            self.optimizer.step()
            accumulated_running_loss_over_all_batches += loss_value.item()
            total_amount_of_samples += targets.size(0)
            n_correct_samples += 4

            pbar.set_postfix({
                'Train Loss': f'{loss_value.item():.4f}',
                'Train Acc': f'{100. * n_correct_samples / total_amount_of_samples:.2f}%'
            })

        return accumulated_running_loss_over_all_batches, total_amount_of_samples, n_correct_samples

    def _train_epoch_forward_gradient_linearRegression(self) -> tuple[float, int, int]:
        accumulated_running_loss_over_all_batches = 0.0
        n_correct_samples = 0
        total_amount_of_samples = 0
        pbar = tqdm(self.data_loader, desc=f'Functional Training Epoch {self.epoch_num}/{self.total_epochs}')

        def loss_fn(params_dict):
            return self._mse_loss(params_dict, inputs, targets)

        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            # initial_state_dict = self.model.state_dict().copy()
            # self.model.load_state_dict(initial_state_dict)
            self.model.train()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            torch.manual_seed(self.seed + batch_idx)
            np.random.seed(self.seed)

            params = dict(self.model.named_parameters())
            loss_value = self._mse_loss(params, inputs, targets)

            for name, param in self.model.named_parameters():
                # v_param = torch.randn_like(param)
                # ToDo: anschauen weleche anderen Verteilungen es noch so gibt
                v_param = torch.randn(param.shape, dtype=param.dtype, device=param.device,
                                      generator=torch.Generator().manual_seed(
                                          self.seed + batch_idx + hash(name) % 10000))
                # print(f' name: {name}\n v_param:\n {v_param}\n')
                v_params_single = {n: torch.zeros_like(p) if n != name else v_param
                                   for n, p in params.items()}
                # print(f' v_params_single for {name}:\n {v_params_single}\n')

                _, directional_derivatives = torch.func.jvp(
                    loss_fn, (params,), (v_params_single,)
                )
                gradient = directional_derivatives * v_param
                param.grad = gradient

            # print(self.model.parameters())
            self.optimizer.step()
            accumulated_running_loss_over_all_batches += loss_value.item()
            total_amount_of_samples += targets.size(0)

            pbar.set_postfix({
                'Train Loss': f'{loss_value.item():.4f}'
            })

        return accumulated_running_loss_over_all_batches, total_amount_of_samples, n_correct_samples

    def _backpropaggation(self) -> tuple[float, int, int]:

        train_losses = []
        acuumulated_running_loss_over_all_Batches = 0.0
        n_correct_samples = 0
        total_amount_of_samples = 0

        pbar = tqdm(iterable=self.data_loader, desc=f'Training Epoch {self.epoch_num}/{self.total_epochs}',
                    bar_format='| {bar} | {desc} -> Batch {n}/{total} | Estimated Time: {remaining} | Time: {elapsed} {postfix}')  # just a nice to have progress bar

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()  # reset gradients from previous iteration

            outputs = self.model(inputs)  # forward pass
            loss = self.loss_function(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())

            # statistics
            acuumulated_running_loss_over_all_Batches += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_amount_of_samples += targets.size(0)
            n_correct_samples += (predicted == targets).sum().item()
            # Update progress bar
            pbar.set_postfix({
                'Train Loss': f' {loss.item():.4f}',
                'Train Acc': f' {100. * n_correct_samples / total_amount_of_samples:.2f}%'
            })
        return  acuumulated_running_loss_over_all_Batches, total_amount_of_samples, n_correct_samples