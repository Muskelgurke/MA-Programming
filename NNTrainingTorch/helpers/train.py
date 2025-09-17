import statistics
import time
import torch
from flax.linen import avg_pool
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def train_epoch(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epoch_num: int,
                total_epochs: int
                )->tuple[float, float]:
    """
            Train the model for one epoch. Visual Feedback in Terminal with tqdm progress bar.
            Gives back Average Train Loss and Train Accuracy.
    """
    model.train() # preparing model fo training

    #ToDo: Name ändern damit man weiß wie viele Batches man hat.
    running_loss, correct, total = backpropaggation(data_loader,
                                                    criterion,
                                                    device,
                                                    model,
                                                    optimizer,
                                                    epoch_num,
                                                    total_epochs)
    avg_train_loss_of_epoch = running_loss / len(data_loader)
    avg_train_acc_of_epoch = 100. * correct / total

    return avg_train_loss_of_epoch, avg_train_acc_of_epoch

def functional_backpropagation(data_loader: torch.utils.data.DataLoader,
                              criterion: nn.Module,
                              device: torch.device,
                              model_function,
                              optimizer: torch.optim.Optimizer,
                              epoch_num: int,
                                epoch_total: int) -> tuple[float, int, int]:
    """
    berechnet die backpprogagation für ein funktionales model (ohne zustand)
    """
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(data_loader, desc=f'Functional Training Epoch {epoch_num}/{epoch_total}')

    def compute_loss(params, inputs, targets):
        # model_function nimmt Parameter + Inputs, gibt Outputs zurück
        outputs = model_function(params, inputs)
        return criterion(outputs, targets)

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Funktionale Gradientenberechnung
        grads = grad_fn(params, inputs, targets)
        jvp * v
        loss = compute_loss(params, inputs, targets)

        # Parameter funktional updaten
        params = functional_optimizer_step(params, grads, optimizer.param_groups[0]['lr'])

        # Statistiken
        running_loss += loss.item()
        outputs = model_function(params, inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        pbar.set_postfix({
            'Train Loss': f'{loss.item():.4f}',
            'Train Acc': f'{100. * correct / total:.2f}%'
        })

    return params, running_loss, correct, total

def functional_optimizer_step(params: dict, grads: dict, lr: float) -> dict:
    """Funktionales Parameter-Update (SGD)"""
    new_params = {}
    for name in params:
        new_params[name] = params[name] - lr * grads[name]
    return new_params

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