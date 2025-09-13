import time
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from NNTrainingJAX.helpers.TrainingData import TrainingData
from NNTrainingTorch.helpers.Config import Config
from NNTrainingTorch.helpers.saving import TorchModelSaver
from NNTrainingJAX.helpers.TrainingResults import TrainingResults
import NNTrainingTorch.helpers.datasets as datasets_helper
import NNTrainingTorch.helpers.model as model_helper


def train_epoch(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                criterion,
                optimizer,
                device: torch.device
                )-> tuple[float, float]:
    model.train() # preparing model fo training
    train_losses = []
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc="Training") # just a nice to have progress bar

    for batch_idx, (inputs,targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad() # reset gradients from previous iteration

        outputs = model(inputs) # forward pass
        loss = criterion(outputs, targets)
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
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    epoch_avg_loss = running_loss / len(train_loader)
    epoch_avg_acc = 100. * correct / total
    return epoch_avg_loss, epoch_avg_acc


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
    criterion = nn.CrossEntropyLoss()

    # Optimizers
    # ToDo: switch between different optimizers
    # ToDo: switch between different learning rates?
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learningRate)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    epoch_times = []

    for epoch in range(config.numEpochs):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_losses)
        test_accs.append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {test_loss:.4f}, Val Acc: {test_acc:.2f}%")


        if config.earlyStopping:
            if epoch > config.patience and test_loss > min(test_losses[-config.patience:]):
                print(f"Early stopping at epoch {epoch + 1}")
                break
        print(f"Epoch {epoch + 1}/{config.numEpochs} completed in {epoch_time:.2f}s")
    print("-"*30)
    print("\nTraining and Testing completed")
    print(f"Best validation accuracy: {max(test_accs):.2f}%")
    print("-" * 30)

    results = TrainingResults(
        train_accs=train_accs,
        test_accs=test_accs,
        train_loss=train_losses,
        test_loss=test_losses,
        epoch_times=epoch_times
    )
    saver = TorchModelSaver()
    saver.save_training_session(training_result=results, config= config, model= model, save_full_model=True)
    #ToDo: irgendwas falsch mit dem Saver- Printed nicht richtig und speicher es nicht ab.



def load_config_File(config_path: str)->Config:
    config = Config.from_yaml(config_path)
    return config

if __name__ == "__main__":
    config_path = "_Configuration/config.yaml"
    training_configurations = load_config_File(config_path)
    train_loader, test_loader = datasets_helper.get_dataloaders(training_configurations)
    start_NN(training_configurations, train_loader, test_loader)




