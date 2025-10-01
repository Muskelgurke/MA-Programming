import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

import numpy as np
from absl.testing.parameterized import named_parameters
from fontTools.misc.timeTools import epoch_diff
from sympy.vector import directional_derivative
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
        y = torch.func.functional_call(self.model, (params, {}), (x,))
        return nn.functional.cross_entropy(y, targets)

    def _cross_entropy_loss_alt(self, params: dict, x: torch.Tensor, targets: torch.Tensor):
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
                running_loss, total, correct = self._backpropaggation()

            case _:
                raise ValueError(f"Unknown Training - Method: {self.config.training_method}")

        avg_train_loss_of_epoch = running_loss / len(self.data_loader)
        avg_train_acc_of_epoch = 100. * correct / total

        return avg_train_loss_of_epoch, avg_train_acc_of_epoch

    def _train_epoch_forward_gradient(self) -> tuple[float, int, int]:
        accumulated_running_loss_over_all_batches = 0
        n_correct_samples = 0
        total_amount_of_samples = 0
        pbar = tqdm(self.data_loader, desc=f'FGD Training Epoch {self.epoch_num}/{self.total_epochs}')

        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            self.model.train()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Get parameters as tuple (like in the example)
            named_params = dict(self.model.named_parameters())
            params = tuple(named_params.values())
            names = tuple(named_params.keys())

            # Sample perturbation vectors for every parameter
            v_params = tuple([torch.randn_like(p) for p in params])

            # Define loss function for functional call
            def loss_fn(params_tuple):
                # Reconstruct parameter dict from tuple
                params_dict = dict(zip(names, params_tuple))
                return self._cross_entropy_loss(params_dict, inputs, targets)
            try:
                # Compute JVP (Forward AD)
                loss, dir_der = torch.func.jvp(loss_fn, (params,), (v_params,))

                # Set gradients: gradient = v * jvp (scalar multiplication)
                with torch.no_grad():
                    for j, param in enumerate(self.model.parameters()):
                        print(f'{v_params[j]=}, {dir_der=}')
                        param.grad =  dir_der * v_params[j]
                        print(param.grad.values())
                        exit()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                # Optimizer step
                self.optimizer.step()

                if torch.isnan(loss):
                    print(f"NaN loss detected in batch {batch_idx}")
                    continue
            except Exception as e:
                print(f'Error in batch {batch_idx}: {e}')
                continue

            # Calculate metrics
            accumulated_running_loss_over_all_batches += loss.item()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_amount_of_samples += targets.size(0)
            n_correct_samples += (predicted == targets).sum().item()
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'Train Loss': f' {loss.item():.4f}',
                'Train Acc': f' {100. * n_correct_samples / total_amount_of_samples:.2f}%'
            })

        pbar.close()
        return (accumulated_running_loss_over_all_batches / len(self.data_loader),
                n_correct_samples,
                total_amount_of_samples)

    def _train_epoch_forward_gradient_4(self) -> tuple[float, int, int]:
        accumulated_running_loss_over_all_batches = 0.0
        n_correct_samples = 0
        total_amount_of_samples = 0
        pbar = tqdm(self.data_loader, desc=f'FGD Training Epoch {self.epoch_num}/{self.total_epochs}')

        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            self.model.train()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            #torch.manual_seed(self.seed + batch_idx)

            # Get parameters as tuple (like in the example)
            named_params = dict(self.model.named_parameters())
            params = tuple(named_params.values())
            names = tuple(named_params.keys())

            # Sample perturbation vectors for every parameter
            v_params = tuple([torch.randn_like(p) for p in params])

            # Define loss function for functional call
            def loss_fn(params_tuple):
                # Reconstruct parameter dict from tuple
                params_dict = dict(zip(names, params_tuple))
                return self._cross_entropy_loss(params_dict, inputs, targets)

            # Compute JVP (Forward AD)
            loss, dir_der = torch.func.jvp(loss_fn, (params,), (v_params,))

            #print(f'{batch_idx=}, {loss=}, {dir_der=}')
            # Set gradients: gradient = v * jvp (scalar multiplication)
            for j, param in enumerate(self.model.parameters()):
                #print(f'{j=}, {param=}')
                param.grad = v_params[j]* dir_der
            #print(dict(self.model.named_parameters()))
            # Optimizer step
            self.optimizer.step()
            #print(dict(self.model.named_parameters()))



            # Calculate metrics
            accumulated_running_loss_over_all_batches += loss.item()

            with torch.no_grad():
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total_amount_of_samples += targets.size(0)
                n_correct_samples += predicted.eq(targets).sum().item()

            pbar.update(1)
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * n_correct_samples / total_amount_of_samples:.2f}%'
            })

        pbar.close()
        return (accumulated_running_loss_over_all_batches / len(self.data_loader),
                n_correct_samples,
                total_amount_of_samples)


    def _train_epoch_forward_gradient_3(self) -> tuple[float, int, int]:
        accumulated_running_loss_over_all_batches = 0.0
        n_correct_samples = 0
        total_amount_of_samples = 0
        pbar = tqdm(self.data_loader, desc=f'FGD Training Epoch {self.epoch_num}/{self.total_epochs}')

        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            self.model.train()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            torch.manual_seed(self.seed + batch_idx)
            np.random.seed(self.seed)

            named_parameters = dict(self.model.named_parameters())
            actual_parameters = {}
            for name, param in named_parameters.items():
                actual_parameters[name] = param.detach()
                #print(f' actual_parameters:\n {actual_parameters}\n')
                submodule_name, parameter_name = name.rsplit(".",1)
                #print(f' submodule_name: {submodule_name}\n parameter_name: {parameter_name}\n')
                submodule = self.model.get_submodule(submodule_name)
                #print(f' submodule:\n {submodule}\n')
                delattr(submodule, parameter_name)
                #print(f' del_submodule:\n {submodule.parameters()}\n')
                setattr(submodule, parameter_name, actual_parameters[name])
                #print(f' new_submodule:\n {submodule.parameters()}\n')

            #print(f'actual Parameters:\t\t\t{actual_parameters}')
            #param_names_and_shapes = {name: param.shape for name, param in actual_parameters.items()}
            #print(f' param_names_and_shapes:\n {param_names_and_shapes}\n')
            #print(f'param_names_and_shapes Values: {list(param_names_and_shapes.values())}\n')
            pertubation_vectors = {}
            for name, param in actual_parameters.items():
               # print(f' name: {name}\n param:\n {param}\n')
                # pertubation_vectors[name] = torch.randn(param.shape,dtype=param.dtype,device=param.device,generator=torch.Generator().manual_seed(self.seed))
                pertubation_vectors[name] = torch.randn_like(param)
                #print(f' pertubation_vectors:\n {pertubation_vectors}\n')

            # Connect the parameters and tangents as an dualTensor for JVP

            estimated_gradients={}
            for param_name in actual_parameters.keys():
                # Reset alle Parameter zu originalen Werten
                """
                for name, param_value in actual_parameters.items():
                    submodule_name, parameter_name = name.rsplit(".", 1)
                    submodule = self.model.get_submodule(submodule_name)
                    delattr(submodule, parameter_name)
                    setattr(submodule, parameter_name, param_value)
"""
                # Schritt 4: Nur für DIESEN Parameter Dual-Tensor erstellen
                with fwAD.dual_level():
                    # Für aktuellen Parameter: Dual-Tensor mit Perturbation
                    submodule_name, parameter_name = param_name.rsplit(".", 1)
                    submodule = self.model.get_submodule(submodule_name)
                    #print(f' submodule for {param_name}:\n {submodule}\n')
                    current_perturbation = pertubation_vectors[param_name]
                    dual_param = fwAD.make_dual(actual_parameters[param_name], current_perturbation)

                    delattr(submodule, parameter_name)
                    setattr(submodule, parameter_name, dual_param)
                    #print(dict(self.model.named_parameters()))
                    # Forward Pass
                    output = self.model(inputs)
                    loss = torch.nn.functional.cross_entropy(output, targets)
                    #print(loss.item())
                    # Schritt 5: Richtungsableitung für diesen Parameter extrahieren
                    directional_derivative = fwAD.unpack_dual(loss).tangent
                    #print(f' Richtungsableitung für {param_name}:\n {directional_derivative}\n')
                    # Schritt 6: Geschätzten Gradienten berechnen
                    # g_estimated = (directional_derivative) * perturbation_vector
                    estimated_gradient = directional_derivative * pertubation_vectors[param_name]
                    estimated_gradients[param_name] = estimated_gradient
                    #print(f' Geschätzter Gradient für {param_name}:\n {estimated_gradient}\n')


            accumulated_running_loss_over_all_batches += loss.item()
            # Schritt 7: Geschätzten Gradienten dem Parameter zuweisen
            with torch.no_grad():
                # set Model parameters to actual values
                for name, param_value in actual_parameters.items():
                    submodule_name, parameter_name = name.rsplit(".", 1)
                    submodule = self.model.get_submodule(submodule_name)
                    if hasattr(submodule, parameter_name):
                        delattr(submodule, parameter_name)
                    # Use nn.Parameter to properly register the parameter
                    setattr(submodule, parameter_name, nn.Parameter(param_value))

                for name, param in self.model.named_parameters():
                    #print(name)
                    if name in estimated_gradients:
                        #gradient zuweisen
                        #print(f'Gradienten zuweisung von: {estimated_gradients[name]}')
                        param.grad = estimated_gradients[name]
            #print(f'Parameter mit Gradienten:\n {list(self.model.named_parameters())}\n')
            #print(f'Parameter Gradients:\n {[param.grad for param in self.model.parameters()]}\n')
            self.optimizer.param_groups[0]['params'] = list(self.model.parameters())
            self.optimizer.step()

            #print (f' loss: {loss}\n')
            # Accuracy berechnen
            _, predicted = output.max(1)
            total_amount_of_samples += targets.size(0)
            n_correct_samples += predicted.eq(targets).sum().item()

            pbar.update(1)
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * n_correct_samples / total_amount_of_samples:.2f}%'
            })

        pbar.close()
        return (accumulated_running_loss_over_all_batches / len(self.data_loader),
                n_correct_samples,
                total_amount_of_samples)

    def _train_epoch_forward_gradient_2(self) -> tuple[float, int, int]:
        """
        Train the model for each epoch using functional forward gradient decent
        :return:
        """
        accumulated_running_loss_over_all_batches = 0.0
        n_correct_samples = 0
        total_amount_of_samples = 0
        pbar = tqdm(self.data_loader, desc=f'FGD Training Epoch {self.epoch_num}/{self.total_epochs}')

        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            self.model.train()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            torch.manual_seed(self.seed + batch_idx)
            np.random.seed(self.seed)

            # Get current parameters as a dictionary
            current_params = dict(self.model.named_parameters())


            # Define loss function using functional_call to avoid in-place operations
            def loss_fn(params_dict):
                return self._cross_entropy_loss(params_dict, inputs, targets)

            # Create random perturbation vectors for each parameter
            param_vectors = {}
            for name, param in self.model.named_parameters():
                v_param = torch.randn(
                    param.shape,
                    dtype=param.dtype,
                    device=param.device,
                    generator=torch.Generator().manual_seed(
                        self.seed + hash(name) % 10000 + batch_idx
                    )
                )
                param_vectors[name] = v_param
            print(f'batch_idx: {batch_idx}\n')
            print(f' param_vectors:\n \t{param_vectors}\n')
            # Compute directional derivative using JVP
            try:
                # JVP returns (primal_out, tangent_out)
                loss, directional_derivative = torch.func.jvp(
                    loss_fn,
                    (current_params,),
                    (param_vectors,)
                )
                print(f' loss: {loss}\n directional_derivative: {directional_derivative}\n')
                # Store accumulated loss
                loss_value = loss
                accumulated_running_loss_over_all_batches += loss_value.item()

                # Compute forward gradient: directional_derivative * v_param
                # Note: directional_derivative is a scalar for the entire loss
                # We need to distribute it to each parameter
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        # Forward gradient approximation
                        gradient_approx = directional_derivative * param_vectors[name]
                        if param.grad is None:
                            param.grad = gradient_approx
                        else:
                            param.grad += gradient_approx

                # Update parameters
                self.optimizer.step()

                # Calculate accuracy
                with torch.no_grad():
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    batch_total = targets.size(0)
                    total_amount_of_samples += batch_total
                    n_correct_samples += (predicted == targets).sum().item()

                # Update progress bar
                current_loss = loss_value.item()
                current_acc = 100. * n_correct_samples / total_amount_of_samples
                pbar.set_postfix({
                    'Train Loss': f'{current_loss:.4f}',
                    'Train Acc': f'{current_acc:.2f}%'
                })
                pbar.update()

            except Exception as e:
                print(f"Error in JVP computation: {e}")
                continue

        pbar.close()
        return accumulated_running_loss_over_all_batches / len(
            self.data_loader), total_amount_of_samples, n_correct_samples

    def _train_epoch_forward_gradient_1_0(self) -> tuple[float, int, int]:
        """
        Train the model for each epoch using functional forward gradient decent
        :return:
        """
        accumulated_running_loss_over_all_batches = 0.0
        n_correct_samples = 0
        total_amount_of_samples = 0
        pbar = tqdm(self.data_loader, desc=f'FGD Training Epoch {self.epoch_num}/{self.total_epochs}')
        """s
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

            param_vectors = {}
            print(f'self.model.named_parameters():\n {list(self.model.named_parameters())}\n')
            for name, param in self.model.named_parameters():
                # Create random vector with same shape as parameter
                v_param = torch.randn(
                    param.shape,
                    dtype=param.dtype,
                    device=param.device,
                    generator=torch.Generator().manual_seed(
                        self.seed + hash(name) % 10000
                    )
                )
                param_vectors[name] = v_param

            print(f' param_vectors:\n {param_vectors}\n')

            for name, param in self.model.named_parameters():
                _,directional_derivatives = torch.func.jvp(
                    loss_fn, (dict(self.model.named_parameters()),), (param_vectors,)
                directional_derivatives = torch.func.jvp(
                    loss_fn, (dict(self.model.named_parameters()),), (param_vectors,)
                )[1])


            exit()
           
            v_param = torch.randn(self.model.named_parameters,
                                    dtype=params.dtype,
                                    device=params.device,
                                    generator=torch.Generator().manual_seed(self.seed + batch_idx))
                #sv_param = torch.randn(param.shape, dtype=param.dtype, device=param.device, generator=torch.Generator().manual_seed(self.seed+batch_idx))
                #print(f' name: {name}\n v_param:\n {v_param}\n')


                #print(f' v_params_single for {name}:\n {v_params_single}\n')
                _,directional_derivatives = torch.func.jvp(
                    loss_fn, (self.model.parameters(),), (v_param,)
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
        accumulated_running_loss_over_all_batches = 0.0
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
            accumulated_running_loss_over_all_batches += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_amount_of_samples += targets.size(0)
            n_correct_samples += (predicted == targets).sum().item()
            # Update progress bar
            pbar.set_postfix({
                'Train Loss': f' {loss.item():.4f}',
                'Train Acc': f' {100. * n_correct_samples / total_amount_of_samples:.2f}%'
            })
        return  accumulated_running_loss_over_all_batches, total_amount_of_samples, n_correct_samples