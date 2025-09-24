import torch
import torch.func
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
import numpy as np

from torch import nn
from tqdm import tqdm


class Trainer:
    """
    A trainer class that uses functional programming approach with JVP for gradient computation.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 total_epochs: int,
                 seed: int):
        """Initialize the trainer with all required parameters."""
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch_num = 0
        self.total_epochs = total_epochs
        self.seed = seed

    def train_epoch(self, epoch_num: int) -> tuple[float, float]:
        """
        Train the model for one epoch using functional forward gradient descent.

        Returns:
            tuple: (average_train_loss, average_train_accuracy)
        """
        """
            # Beispiel von Leo wie man Gradienten zuordnen kann

            model.train()  # preparing model fo training
            model.zero_grad()

            for param in model.parameters():
                param.grad = torch.ones_like(param)

            for param in model.parameters():
                print(f"{param=}")
                print(f"{param.grad=}")
            optimizer.step()
            print("OPTIM STEP")
            for param, grad in zip(model.parameters(), grads):
                param.grad = grad

            exit()
        """
        self.epoch_num = epoch_num
        self.model.train()

        #ToDo: hier muss noch eine switch case rein damit man zwischen backprop und FGD switchen kann
        # Add deterministic
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        #running_loss, correct, total = self._functional_forward_gradient_descent_1()
        running_loss, correct, total = self._train_epoch_forward_gradient()

        avg_train_loss_of_epoch = running_loss / len(self.data_loader)
        avg_train_acc_of_epoch = 100. * correct / total

        return avg_train_loss_of_epoch, avg_train_acc_of_epoch

    def _functional_loss(self,
                         params: dict,
                         buffers: dict,
                         inputs: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
        """Compute functional loss using torch.func.functional_call."""
        outputs = torch.func.functional_call(self.model, (params, buffers), inputs)
        return self.criterion(outputs, targets)

    def _train_epoch_forward_gradient(self) -> tuple[float, int, int]:
        accumulated_running_loss_over_all_batches = 0.0
        n_correct_samples = 0
        total_amount_of_samples = 0

        self.model.train()
        #ToDo: gibt es vielleicht auch ohne numpy einen RNG ?


        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            inputs,targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            rng = np.random.default_rng(self.seed)
            tangent_np = rng.normal(size=inputs.shape)
            tangent = torch.from_numpy(tangent_np).float().to(self.device)
            print(f' inputs:\n {inputs}\n tangents: \n {tangent}')
            jvps = []
            with fwAD.dual_level():
                dual_inputs = fwAD.make_dual(inputs, tangent)
                print(f'dual_inputs: {dual_inputs}')
                output = self.model(dual_inputs)
                loss = F.mse_loss(output, targets)
                print(loss)
                #print(fwAD.unpack_dual(loss).tangent)
                jvps.append(fwAD.unpack_dual(loss).tangent)
                print(f' jvps: {jvps}')
        return accumulated_running_loss_over_all_batches, total_amount_of_samples, n_correct_samples


    def _functional_forward_gradient_descent_1(self) -> tuple[float, int, int]:
        """
        Perform functional forward gradient descent using JVP.
        """
        accumulated_running_loss_over_all_batches = 0.0
        n_correct_samples = 0
        total_amount_of_samples = 0
        pbar = tqdm(self.data_loader, desc=f'Functional Training Epoch {self.epoch_num}/{self.total_epochs}')

        params_dict = dict(self.model.named_parameters())
        buffers_dict = dict(self.model.named_buffers())
        params_tuple = tuple(params_dict.values())

        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # Initialize perturbation vector
            v_params = tuple([torch.randn_like(p) for p in params_tuple])

            loss_fn = lambda params: self._functional_loss(
                dict(zip(params_dict.keys(), params)), buffers_dict, inputs, targets
            )

            # Compute JVP
            loss, directional_derivatives = torch.func.jvp(loss_fn, (params_tuple,), (v_params,))

            # Set gradients using directional derivatives
            for i, param in enumerate(self.model.parameters()):
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad = directional_derivatives * v_params[i]

            self.optimizer.step()

            accumulated_running_loss_over_all_batches += loss.item()

            with torch.no_grad():
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_amount_of_samples += targets.size(0)
                n_correct_samples += (predicted == targets).sum().item()

            pbar.set_postfix({
                'Train Loss': f'{loss.item():.4f}',
                'Train Acc': f'{100. * n_correct_samples / total_amount_of_samples:.2f}%'
            })

        return accumulated_running_loss_over_all_batches, n_correct_samples, total_amount_of_samples