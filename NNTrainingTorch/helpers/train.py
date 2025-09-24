import torch
import torch.func

import numpy as np

from torch import nn

from tqdm import tqdm


def train_epoch(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epoch_num: int,
                total_epochs: int
                ) -> tuple[float, float]:
    """
            Train the model for one epoch. Visual Feedback in Terminal with tqdm progress bar.
            Gives back Average Train Loss and Train Accuracy.
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

    #ToDo: Name ändern damit man weiß wie viele Batches man hat.
    running_loss, correct, total = functional_forward_gradient_decent(data_loader,
                                                                      criterion,
                                                                      device,
                                                                      model,
                                                                      optimizer,
                                                                      epoch_num,
                                                                      total_epochs)
    """
    running_loss, correct, total = backpropagation(data_loader,
                                                   criterion,
                                                   device,
                                                   model,
                                                   optimizer,
                                                   epoch_num,
                                                   total_epochs)
    """

    avg_train_loss_of_epoch = running_loss / len(data_loader)
    avg_train_acc_of_epoch = 100. * correct / total

    return avg_train_loss_of_epoch, avg_train_acc_of_epoch

def functional_loss(params: dict,
                    buffers: dict,
                    model: nn.Module,
                    inputs: torch.Tensor,
                    targets: torch.Tensor,
                    criterion: nn.Module) -> torch.Tensor:

    outputs = torch.func.functional_call(model, (params, buffers), inputs)

    return criterion(outputs, targets)


def functional_forward_gradient_decent(
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device,
        model,
        optimizer: torch.optim.Optimizer,
        epoch_num: int,
        epoch_total: int
    ) -> tuple[float, int, int]:
        """
        Berechnet die Backpropagation für ein funktionales Model (ohne Zustand)
        """

        acuumulated_running_loss_over_all_Batches = 0.0
        n_correct_samples = 0
        total_amount_of_samples = 0
        pbar = tqdm(data_loader, desc=f'Functional Training Epoch {epoch_num}/{epoch_total}')

        params_dict = dict(model.named_parameters())
        # Braucht man für die korrekte Ausführung von BatchNorm und Dropout. Wie genau ist noch ein Rätsel!!
        buffers_dict = dict(model.named_buffers())
        params_tuple = tuple(params_dict.values())

        # buffers = named_buffers.keys()
        # print(f"Initial weight:{model.linear.weight}")
        # print(f"Initial bias:{model.linear.bias}")

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Reset Gradients to zero

            # Pertubation Vektor initialisieren
            v_params = tuple([torch.randn_like(p) for p in params_tuple])

            loss_fn = lambda params: functional_loss(
                dict(zip(params_dict.keys(), params)), buffers_dict, model, inputs, targets, criterion
            )

            # JVP berechnen
            loss, directional_derivatives = torch.func.jvp(loss_fn, (params_tuple,), (v_params,))

            # directional_derivative ist bereits ∇f(x)·v (ein Skalar)
            # Wir müssen die Gradienten entsprechend setzen
            for i, param in enumerate(model.parameters()):
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad = directional_derivatives * v_params[i]

            optimizer.step()

            acuumulated_running_loss_over_all_Batches += loss.item()
            with torch.no_grad():
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_amount_of_samples += targets.size(0)
                n_correct_samples += (predicted == targets).sum().item()

        return acuumulated_running_loss_over_all_Batches, n_correct_samples, total_amount_of_samples

def functional_optimizer_step(params: dict, grads: dict, lr: float) -> dict:
     """Funktionales Parameter-Update (SGD)
        mit Momentum
     """
     new_params = {}
     momentum = []
     for name in params:
        grads[name] += 0.9 * momentum[name]
        momentum[name] = grads[name]
        new_params[name] = params[name] - lr * grads[name]
     return new_params

def backpropagation(data_loader: torch.utils.data.DataLoader,
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
    return acuumulated_running_loss_over_all_Batches, n_correct_samples, total_amount_of_samples