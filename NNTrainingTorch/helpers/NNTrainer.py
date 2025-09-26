import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

import numpy as np
from absl.testing.parameterized import named_parameters

from torch import nn
from torch.autograd.forward_ad import unpack_dual
from tqdm import tqdm
from torch.func import functional_call


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

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _mse_loss(self, params: dict, x: torch.Tensor, targets: torch.Tensor):
        model = lambda p, x: torch.func.functional_call(self.model, (p, {}), x)
        y = model(params, x)
        return nn.functional.mse_loss(y, targets)

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

        #ToDo: Backpropagation als alternative Methode einbauen bzw. wechseln je nach Config
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
        pbar = tqdm(self.data_loader, desc=f'Functional Training Epoch {self.epoch_num}/{self.total_epochs}')

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

            def loss_fn(params_dict):
                return self._mse_loss(params_dict, inputs, targets)

            for name, param in self.model.named_parameters():
                #v_param = torch.randn_like(param)
                #ToDo: anschauen weleche anderen Verteilungen es noch so gibt
                v_param=torch.randn(param.shape, dtype=param.dtype, device=param.device,
                     generator=torch.Generator().manual_seed(self.seed + batch_idx + hash(name) % 10000))
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


        """
        params = {name: p for name, p in self.model.named_parameters()}

        tangents = {name: torch.rand_like(p) for name, p in params.items()}

        print(f' params:\n {params}\n')
        print(f' tangents:\n {tangents}\n')
        dual_params= {}
        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            with fwAD.dual_level():
                for name, p in params.items():
                    print(f'Named_parameters : {self.model.named_parameters()}')
                    dual_params[name] = fwAD.make_dual(p, tangents[name])
                    
                out = functional_call(self.model, dual_params, inputs)
                jvp = fwAD.unpack_dual(out).tangent
                print(f'\t\t\t\t\t output: {out}\n')
                print(f'\t\t\t\t\tjvp: {jvp}\n')

        


        
        self.model.load_state_dict(initial_state_dict)
        self.model.train()
        inputs,targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        torch.manual_seed(self.seed + batch_idx)
        np.random.seed(self.seed)





        #### ---- Forward AD --- ####sss
        with fwAD.dual_level():
            for param in self.model.parameters():
                tangent = torch.randn_like(param, device=self.device)
                print(f' tangent:\n {tangent}\n')
                param.data = fwAD.make_dual(param.data,tangent)
                print(f' params_dual:\n {param.data}\n')
                print(f'---*30---')
            output = self.model(inputs)
            loss = F.mse_loss(output, targets)
            print(f'My loss:\n {loss}\n')
            #print(fwAD.unpack_dual(loss).tangent)
            #y, jvp = unpack_dual(loss)
            #print(f' y: {y}\n jvp: {jvp}\n')
            loss_value, loss_jvp = fwAD.unpack_dual(loss)

            print(f' loss_value: {loss_value}\n loss_jvp: {loss_jvp}\n')

            # clip gradients
            for param in self.model.parameters():
                param_value, param_jvp = fwAD.unpack_dual(param.data)
                param.data = param_value
                param.grad = param_jvp
            """

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

def backpropaggation(data_loader: torch.utils.data.DataLoader,
                     criterion: nn.Module,
                     device: torch.device,
                     model: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     epoch_num: int,
                    epoch_total: int) -> tuple[float, int, int]:
    train_losses = []
    acuumulated_running_loss_over_all_Batches = 0.0
    n_correct_samples = 0
    total_amount_of_samples = 0

    pbar = tqdm(iterable=data_loader, desc=f'Training Epoch {epoch_num}/{epoch_total}',
                bar_format='| {bar} | {desc} -> Batch {n}/{total} | Estimated Time: {remaining} | Time: {elapsed} {postfix}')  # just a nice to have progress bar

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # reset gradients from previous iteration

        outputs = model(inputs)  # forward pass
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
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
    return  acuumulated_running_loss_over_all_Batches, n_correct_samples, total_amount_of_samples