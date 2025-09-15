import statistics
import time
import torch
from flax.linen import avg_pool
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from NNTrainingTorch.helpers.config_class import Config
from NNTrainingTorch.helpers.saver_class import TorchModelSaver
from NNTrainingTorch.helpers.TrainingResults import TrainingResults
import NNTrainingTorch.helpers.datasets as datasets_helper
import NNTrainingTorch.helpers.model as model_helper


def train_epoch(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                loss_function: nn.Module,
                optimizer,
                device: torch.device,
                epoch_num: int,
                total_epochs: int
                )->tuple[float, float]:
    model.train() # preparing model fo training
    train_losses = []
    running_loss = 0.0
    correct = 0
    total = 0
    #ToDo: Name ändern damit man weiß wie viele Batches man hat.
    pbar = tqdm(iterable =train_loader, desc=f'Training Epoch {epoch_num}/{total_epochs}',
                bar_format='{desc}: {percentage:3.0f}%|{bar}| Estimated Time: {remaining} {postfix}') #just a nice to have progress bar
    for batch_idx, (inputs,targets) in enumerate(pbar):
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad() # reset gradients from previous iteration

        outputs = model(inputs) # forward pass
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        # statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{ loss.item():.4f}',
            'Acc': f'{ 100. * correct / total:.2f}%'
        })

    avg_train_loss_of_epoch = running_loss / len(train_loader)
    avg_train_acc_of_epoch = 100. * correct / total
    return avg_train_loss_of_epoch, avg_train_acc_of_epoch


def test_epoch(model, test_loader, criterion, device):
    """
        Validate the model on test data.

        Why needed: Separate validation prevents data leakage and monitors overfitting.
        """
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for efficiency
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            test_loss += criterion(outputs, targets).item()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


def start_NN(config: Config, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model_helper.get_model(config)
    model.to(device)

    # Loss function
    # ToDo: switch between different loss functions
    loss_function = nn.CrossEntropyLoss()

    # Optimizers
    # ToDo: switch between different optimizers
    # ToDo: switch between different learning rates?
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    epoch_times = []

    print("Training gestartet...")
    print("-" * 60)
    print(" Schritt für Schritt ")

    for epoch in range(config.num_epochs):
        start_time = time.time()
        train_loss_of_Epoch, train_acc_of_Epoch = train_epoch(model,
                                                              train_loader,
                                                              loss_function,
                                                              optimizer,
                                                              device,
                                                              epoch+1,
                                                              config.num_epochs)

        test_loss_of_Epoch, test_acc_of_Epoch = test_epoch(model,
                                                           test_loader,
                                                           loss_function,
                                                           device)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        train_losses.append(train_loss_of_Epoch)
        train_accs.append(train_acc_of_Epoch)
        test_losses.append(test_loss_of_Epoch)
        test_accs.append(test_acc_of_Epoch)

        if config.early_stopping:
            if epoch > config.early_stopping_patience and test_loss_of_Epoch > min(test_losses[-config.early_stopping_patience:]):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        '''
        print(f"Epoch {epoch + 1:2d}/{config.num_epochs} | "
              f"Zeit: {epoch_time:5.2f}s | "
              f"Train Acc: {train_acc_of_Epoch:.4f} | "
              f"Test Acc: {test_acc_of_Epoch:.4f} | "
              f"Train Loss: {train_loss_of_Epoch:.4f} | "
              f"Test Loss: {test_loss_of_Epoch:.4f}")
        '''

    print("-"* 60)
    print("\nTraining and Testing completed")
    print(f"Training abgeschlossen! Durchschnittliche Zeit pro Epoch: {statistics.mean(epoch_times):.2f}s")
    print("-" * 60)

    results = TrainingResults(
        train_accs=train_accs,
        test_accs=test_accs,
        train_losses=train_losses,
        test_losses=test_losses,
        final_params= '',
        epoch_times= epoch_times
    )
    saver = TorchModelSaver()
    saver.save_training_session(training_result=results, config= config, model= model, save_full_model=True)
    #ToDo: irgendwas falsch mit dem Saver- Printed nicht richtig und speicher es nicht ab.



def load_config_File(config_path: str)->Config:
    return Config.from_yaml(config_path)

if __name__ == "__main__":
    config_path = "_Configuration/config.yaml"
    training_configurations = load_config_File(config_path)
    train_loader, test_loader = datasets_helper.get_dataloaders(training_configurations)
    start_NN(training_configurations, train_loader, test_loader)




