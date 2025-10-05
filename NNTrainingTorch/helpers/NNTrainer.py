import torch
import torch.autograd.forward_ad as fwAD
import numpy as np
import csv
import os
from pathlib import Path
from torch import nn
from tqdm import tqdm
from NNTrainingTorch.helpers.config_class import Config
from torch.utils.tensorboard import SummaryWriter
from NNTrainingTorch.helpers.Training_Metriken import TrainingMetrics

class Trainer:
    """
    A trainer class that uses functional programming approach with JVP for gradient computation.
    """

    def __init__(self,
                 config_file: Config,
                 model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 loss_function: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 total_epochs: int,
                 seed: int,
                 tensorboard_writer: torch.utils.tensorboard.SummaryWriter):
        """Initialize the trainer with all required parameters."""
        self.model = model
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.epoch_num = 0
        self.total_epochs = total_epochs
        self.seed = seed
        self.config = config_file

        # Logging
        self.writer = tensorboard_writer
        self.training_dir = self.writer.log_dir

        self.metrics = TrainingMetrics()
        self.metrics.cosine_similarities_of_estgrad_gradient_for_each_batch = []
        self.metrics.estimated_gradients = []

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


    def _cross_entropy_loss(self, params: dict, x: torch.Tensor, targets: torch.Tensor):
        y = torch.func.functional_call(self.model, (params, {}), (x,))
        return nn.functional.cross_entropy(y, targets)


    def _calculate_gradient_metrics(self,
                                    estimated_grads_flat: torch.Tensor,
                                    true_grads_flat: torch.Tensor) -> None:
        # Distance-based metrics
        self.metrics.mse_true_esti_grads = nn.MSELoss()(estimated_grads_flat, true_grads_flat).item()
        self.metrics.mae_true_esti_grad = nn.L1Loss()(estimated_grads_flat, true_grads_flat).item()

        # Standard deviation metrics
        gradient_diff = estimated_grads_flat - true_grads_flat
        self.metrics.std_of_difference_true_esti_grads = torch.std(gradient_diff).item()

        self.metrics.std_of_esti_grads = torch.std(estimated_grads_flat).item()
        self.metrics.std_of_true_grads = torch.std(true_grads_flat).item()

        # Cosine similarity
        self.metrics.cosine_sim_for_batch = torch.nn.functional.cosine_similarity(
            true_grads_flat.unsqueeze(0),
            estimated_grads_flat.unsqueeze(0),
            dim=1
        ).item()


        self.metrics.cosine_similarities_of_estgrad_gradient_for_each_batch.append(
            self.metrics.cosine_sim_for_batch
        )



    def train_epoch(self, epoch_num: int):
        """
        Train the model for one epoch using functional forward gradient descent.
        """

        self.epoch_num = epoch_num + 1
        self.model.train()

        match self.config.training_method:
            case "fgd":
                if self.config.dataset_name == "demo_linear_regression":
                    self._train_epoch_forward_gradient_linearRegression()
                else:
                    self._train_epoch_forward_gradient_and_true_gradient()
                    #self._train_epoch_forward_gradient()
            case "bp":
                self._backpropaggation()
            case _:
                raise ValueError(f"Unknown Training - Method: {self.config.training_method}")

        return self.metrics

    def _train_epoch_forward_gradient_and_true_gradient(self) -> None:
        accumulated_running_loss_over_all_batches = 0
        acc_of_all_batches = []
        accumulated_correct_samples = 0
        accumulated_total_samples = 0
        n_correct_samples = 0
        total_amount_of_samples = 0

        pbar = tqdm(self.data_loader, desc=f'FGD Training Epoch {self.epoch_num}/{self.total_epochs}')

        for batch_idx, (inputs, targets) in enumerate(self.data_loader):

            # data loading
            self.model.train()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # Forward

            # Get parameters as tuple (like in the example)
            named_params = dict(self.model.named_parameters())
            params = tuple(named_params.values())
            names = tuple(named_params.keys())

            # Sample perturbation vectors for every parameter
            v_params = tuple([torch.randn_like(p) for p in params])

            # Define loss function for functional call
            def loss_fn(params_tuple, inputs, targets):
                # Reconstruct parameter dict from tuple
                params_dict = dict(zip(names, params_tuple))
                output = torch.func.functional_call(self.model, params_dict, inputs)
                return nn.functional.cross_entropy(output, targets)

            try:
                # calculate true gradient with backprop
                with torch.enable_grad():
                    outputs_true = self.model(inputs)
                    loss_true = nn.functional.cross_entropy(outputs_true, targets)

                    true_grads = torch.autograd.grad(outputs=loss_true,
                                                     inputs= self.model.parameters(),
                                                     create_graph=False,
                                                     retain_graph=False)
                    true_grads_flat = torch.cat([g.view(-1) for g in true_grads])

                # Compute JVP (Forward AD)
                loss, dir_der = torch.func.jvp(
                    lambda params: loss_fn(params, inputs, targets),
                    (params,),
                    (v_params,)
                )
                estimated_gradients = []
                # Set gradients: gradient = v * jvp (scalar multiplication)
                with torch.no_grad():
                    for j, param in enumerate(self.model.parameters()):
                        estimated_grad = dir_der * v_params[j]
                        param.grad = estimated_grad
                        estimated_gradients.append(estimated_grad)

                estimated_grads_flat = torch.cat([g.view(-1) for g in estimated_gradients])


                self._calculate_gradient_metrics(estimated_grads_flat, true_grads_flat)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                self.optimizer.step()

                if torch.isnan(loss):
                    print(f"NaN loss detected in batch {batch_idx}")

            except Exception as e:
                print(f'Error in batch {batch_idx}: {e}')

            # Calculate metrics
            accumulated_running_loss_over_all_batches += loss.item()

            with torch.no_grad():
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_amount_of_samples += targets.size(0)
                n_correct_samples += (predicted == targets).sum().item()
                acc_of_batch = 100. * n_correct_samples / total_amount_of_samples

            acc_of_all_batches.append(acc_of_batch)
            accumulated_correct_samples += n_correct_samples
            accumulated_total_samples += total_amount_of_samples

            # Tensorboard Logging
            unique_increasing_counter = (self.epoch_num - 1) * len(self.data_loader) + batch_idx
            if self.writer is not None:
                self.writer.add_scalar('Train/Loss', loss.item(), unique_increasing_counter)
                self.writer.add_scalar('Train/Accuracy', acc_of_batch, unique_increasing_counter)
                self.writer.add_scalar('Train/Cosine_Similarity_EstGrad_TrueGrad',
                                       self.metrics.cosine_sim_for_batch,
                                       unique_increasing_counter)
                self.writer.add_scalar('GradientMetrics/STD_Difference',
                                       self.metrics.std_of_difference_true_esti_grads,
                                       unique_increasing_counter)
                self.writer.add_scalar('GradientMetrics/STD_Estimated',
                                       self.metrics.std_of_esti_grads,
                                       unique_increasing_counter)
                self.writer.add_scalar('GradientMetrics/STD_True',
                                       self.metrics.std_of_true_grads,
                                       unique_increasing_counter)
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'Train Loss': f' {loss.item():.4f}'
            })

        # Calculate Train Metrics
        self.metrics.accumulated_total_samples_of_all_batches = accumulated_total_samples
        self.metrics.accumulated_correct_samples_of_all_batches = accumulated_correct_samples

        self.metrics.avg_train_acc_of_epoch = float(np.mean(acc_of_all_batches))

        self.metrics.train_acc = (
                accumulated_correct_samples / accumulated_total_samples * 100
        )

        self.metrics.avg_train_loss_of_epoch = (
                accumulated_running_loss_over_all_batches / len(self.data_loader)
        )

        self.metrics.avg_cosine_similarity_of_epoch = float(np.mean(
            self.metrics.cosine_similarities_of_estgrad_gradient_for_each_batch
        ))


        pbar.close()

    def _train_epoch_forward_gradient(self) -> None:
        accumulated_running_loss_over_all_batches = 0
        acc_of_all_batches = []
        pbar = tqdm(self.data_loader, desc=f'FGD Training Epoch {self.epoch_num}/{self.total_epochs}')

        for batch_idx, (inputs, targets) in enumerate(self.data_loader):

            # data loading
            self.model.train()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # Forward

            #Get parameters as tuple (like in the example)
            named_params = dict(self.model.named_parameters())
            params = tuple(named_params.values())
            names = tuple(named_params.keys())

            # Sample perturbation vectors for every parameter
            v_params = tuple([torch.randn_like(p) for p in params])

            # Define loss function for functional call
            def loss_fn(params_tuple, inputs, targets):
                # Reconstruct parameter dict from tuple
                params_dict = dict(zip(names, params_tuple))
                output = torch.func.functional_call(self.model, params_dict, inputs)
                return  nn.functional.cross_entropy(output, targets)

            try:
                # Compute JVP (Forward AD)
                loss, dir_der = torch.func.jvp(
                    lambda params: loss_fn(params, inputs, targets),
                    (params,),
                    (v_params,)
                )

                # Set gradients: gradient = v * jvp (scalar multiplication)
                with torch.no_grad():
                    for j, param in enumerate(self.model.parameters()):
                        #print(f'{self.model.named_parameters()}')
                        #print(f'{v_params[j]=}, {dir_der=}')
                        param.grad =  dir_der * v_params[j]
                        #print(param.grad)

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

            acc_of_batch = self._calculation_acc_of_batch(inputs,targets)
            acc_of_all_batches.append(acc_of_batch)

            # Tensorboard Logging
            unique_increasing_counter = (self.epoch_num - 1) * len(self.data_loader) + batch_idx
            if self.writer is not None:
                self.writer.add_scalar('Train/Loss', loss.item(), unique_increasing_counter)
                self.writer.add_scalar('Train/Accuracy', acc_of_batch, unique_increasing_counter)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'Train Loss': f' {loss.item():.4f}'
            })

        # Calculate Train Metrics

        self.avg_train_acc_of_epoch = np.mean(acc_of_all_batches)

        self.train_acc = self.accumulated_correct_samples_of_all_batches / self.accumulated_total_samples_of_all_batches * 100

        self.avg_train_loss_of_epoch = accumulated_running_loss_over_all_batches / len(self.data_loader)

        pbar.close()

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

    def _backpropaggation(self) -> None:
        accumulated_running_loss_over_all_batches = 0
        acc_of_all_batches = []

        pbar = tqdm(self.data_loader, desc=f'BP - Training Epoch {self.epoch_num}/{self.total_epochs}') # just a nice to have progress bar

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()  # reset gradients from previous iteration

            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            loss.backward()
            self.optimizer.step()
            accumulated_running_loss_over_all_batches += loss.item()

            # statistics
            acc_of_batch = self._calculation_acc_of_batch(inputs, targets)
            acc_of_all_batches.append(acc_of_batch)

            # Tensorboard Logging
            unique_increasing_counter = (self.epoch_num - 1) * len(self.data_loader) + batch_idx
            if self.writer is not None:
                self.writer.add_scalar('Train/Loss', loss.item(), unique_increasing_counter)
                self.writer.add_scalar('Train/Accuracy', acc_of_batch, unique_increasing_counter)

            # Update progress bar
            pbar.set_postfix({
                'Train Loss': f' {loss.item():.4f}',
                'Train Acc': f' {acc_of_batch:.2f}%'
            })

        self.avg_train_acc_of_epoch = np.mean(acc_of_all_batches)

        self.train_acc = self.accumulated_correct_samples_of_all_batches / self.accumulated_total_samples_of_all_batches * 100

        self.avg_train_loss_of_epoch = accumulated_running_loss_over_all_batches / len(self.data_loader)

