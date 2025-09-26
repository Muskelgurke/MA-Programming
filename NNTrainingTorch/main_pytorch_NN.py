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

from NNTrainingTorch.helpers.NNTrainer import Trainer
import NNTrainingTorch.helpers.datasets as datasets_helper
import NNTrainingTorch.helpers.model as model_helper





def test_epoch(model: torch.nn.Module,
               test_loader: torch.utils.data.DataLoader,
               criterion: nn.Module,
               device: torch.device,
               epoch_num: int,
               total_epochs:int) -> tuple[float, float]:
    """
        Validate the model on test data for one epoch. Visual Feedback in Terminal with tqdm progress bar.
        Gives back Average Test Loss and Test Accuracy.

    """
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(iterable=test_loader, desc=f'Test Epoch {epoch_num}/{total_epochs}',
                bar_format='| {bar} | {desc} -> Batches{total} {postfix}')
    with torch.no_grad():  #Disable gradient computation for efficiency
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            test_loss += criterion(outputs, targets).item()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Test Acc': f'{current_acc:.2f}%'
            })

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total


    return test_loss, test_acc


def start_NN(config: Config, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Loss function

    model = model_helper.get_model(config)
    model.to(device)
    print("-" * 60)
    print(model)
    print("-" * 60)
    # Optimizers
    # ToDo: switch between different learning rates?
    if config.dataset_name == "demo_linear_regression":

        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        loss_function = nn.MSELoss()

    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        loss_function = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    epoch_times = []
    best_loss = float('inf')
    patience_counter = 0

    print("Training gestartet...")
    print(f"\tDataset:\t{config.dataset_name}\n\tModel:\t{config.model_type}")
    print("-" * 60)
    print(" Schritt für Schritt ")

    trainer = Trainer(config,model,train_loader,loss_function,optimizer,device,config.num_epochs,config.random_seed)

    for epoch in range(config.num_epochs):
        start_time = time.time()

        train_loss_of_Epoch, train_acc_of_Epoch = trainer.train_epoch(epoch+1)

        if test_loader is not None:
            test_loss_of_Epoch, test_acc_of_Epoch = test_epoch(model,
                                                           test_loader,
                                                           loss_function,
                                                           device,
                                                           epoch+1,
                                                           config.num_epochs)
        else:
            test_loss_of_Epoch, test_acc_of_Epoch = 0, 0
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        train_losses.append(train_loss_of_Epoch)
        train_accs.append(train_acc_of_Epoch)
        test_losses.append(test_loss_of_Epoch)
        test_accs.append(test_acc_of_Epoch)

        if config.early_stopping:
            if train_loss_of_Epoch < best_loss:
                best_loss = train_loss_of_Epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(
                        f"\n\tStopping early at Epoch {epoch + 1} - No improvement for {config.early_stopping_patience} epochs\n")
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
    print(f"Durchschnittliche Zeit pro Epoch: {statistics.mean(epoch_times):.2f}s")
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

    ###########################################################################################
    ###########################################################################################
    # Linear Regresssion Demo
    ###########################################################################################

    if config.dataset_name == "linear_regression":
        print(f'\nYour New Funktion: y = {model.linear.weight.item():.6f} * X + {model.linear.bias.item():.6f}\n')
    #weights, bias = model.get_weights_and_bias()
    #print(f'Numpy - FG with Auto Grad Lib\t y = {weights} * X + {bias}\n')
    #ToDo: irgendwas falsch mit dem Saver- Printed nicht richtig und speicher es nicht ab.



def load_config_File(config_path: str)->Config:
    return Config.from_yaml(config_path)

if __name__ == "__main__":
    config_path = "_Configuration/config.yaml"

    training_configurations = load_config_File(config_path)

    train_loader, test_loader = datasets_helper.get_dataloaders(training_configurations)

    start_NN(training_configurations, train_loader, test_loader)




