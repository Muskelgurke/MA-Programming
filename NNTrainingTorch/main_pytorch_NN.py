import datetime
import statistics
import time
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from NNTrainingTorch.helpers.Training_Metriken import TrainingMetrics
from NNTrainingTorch.helpers.config_class import Config
from NNTrainingTorch.helpers.saver_class import TorchModelSaver
from NNTrainingTorch.helpers.results_of_epochs import results_of_epochs
from NNTrainingTorch.helpers.EarlyStopping import EarlyStopping
from NNTrainingTorch.helpers.NNTrainer import Trainer
from NNTrainingTorch.helpers.NNTester import Tester
import NNTrainingTorch.helpers.datasets as datasets_helper
import NNTrainingTorch.helpers.model as model_helper

def start_NN(config: Config, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model_helper.get_model(config)
    model.to(device)
    print("-" * 60)
    print(model)
    print("-" * 60)
    print(f"\t Dataset:\t{config.dataset_name}\n\tModel:\t{config.model_type}")
    print("-" * 60)
    print("\t Schritt für Schritt ")

    loss_function, optimizer = get_optimizer_and_lossfunction(config=config,
                                                              model=model)

    early_stopping = EarlyStopping(patience=config.early_stopping_patience,
                                   delta=config.early_stopping_delta) if config.early_stopping else None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = f'runs/{config.dataset_name}_{config.model_type}_{config.training_method}_{timestamp}'
    writer = SummaryWriter(log_dir=run_path)
    saver = TorchModelSaver(writer.log_dir) # eigener Saver gebaut. Plotting etc.

    trainer = Trainer(config_file=config,
                      model=model,
                      data_loader=train_loader,
                      loss_function=loss_function,
                      optimizer=optimizer,
                      device=device,
                      total_epochs=config.epoch_num,
                      seed=config.random_seed,
                      tensorboard_writer=writer)

    tester = Tester(config_file=config,
                    model=model,
                    test_loader=test_loader,
                    loss_function=loss_function,
                    device=device,
                    total_epochs=config.epoch_num,
                    seed=config.random_seed,
                    tensorboard_writer=writer)


    train_losses_per_epoch = []
    train_accs_per_epoch = []
    test_losses_per_epoch = []
    test_accs_per_epoch = []
    epoch_times_per_epoch = []
    training_results_per_epoch = []
    training_results = TrainingMetrics()

    for epoch in range(config.epoch_num):
        start_time = time.time()

        training_results = trainer.train_epoch(epoch_num=epoch)

        tester.validate_epoch(epoch_num=epoch)

        epoch_time = time.time() - start_time
        epoch_times_per_epoch.append(epoch_time)


        train_losses_per_epoch.append(training_results.epoch_avg_train_loss)
        train_accs_per_epoch.append(training_results.train_acc_of_epoch)
        test_losses_per_epoch.append(tester.avg_validation_loss_of_epoch)
        test_accs_per_epoch.append(tester.avg_validation_acc_of_epoch)
        training_results_per_epoch.append(training_results)

        if writer is not None:
            writer.add_scalar('Train/Loss - Epoch',
                              training_results.epoch_avg_train_loss,
                              epoch)
            writer.add_scalar('Train/Accuracy - Epoch',
                              training_results.train_acc_of_epoch)
            writer.add_scalar('Test/Loss - Epoch',
                              tester.avg_validation_loss_of_epoch,
                              epoch)
            writer.add_scalar('Test/Accuracy - Epoch',
                              tester.avg_validation_acc_of_epoch,
                              epoch)

        saver.save_training_metrics_to_csv(training_results,config,epoch)
        early_stopping(tester.avg_validation_loss_of_epoch, model)
        if early_stopping.early_stop:
            print(f"Stopping early at Epoch {epoch + 1}")
            break

    print("-"* 60)
    print("\nTraining and Testing completed")
    print(f"Durchschnittliche Zeit pro Epoch: {statistics.mean(epoch_times_per_epoch):.2f}s")
    print("-" * 60)

    results = results_of_epochs(
        train_accs  =     train_accs_per_epoch,
        test_accs   =      test_accs_per_epoch,
        train_losses=   train_losses_per_epoch,
        test_losses =    test_losses_per_epoch,
        epoch_times =    epoch_times_per_epoch,
        training_results = training_results_per_epoch
    )

    saver.save_training_session(results_of_epoch=results,
                                config= config,
                                model= model,
                                save_full_model=True)


    ###########################################################################################
    # Linear Regresssion Demo Extra Output
    if config.dataset_name == "demo_linear_regression":
        print(f'\nYour New Funktion: y = {model.linear.weight.item():.6f} * X + {model.linear.bias.item():.6f}\n')
    ###########################################################################################


def get_optimizer_and_lossfunction(config: Config, model: torch.nn.Module) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    loss_function = None
    optimizer = None
    match config.dataset_name:
        case "demo_linear_regression":
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
            loss_function = nn.MSELoss()
        case "mnist" | "fashionmnist" | "small_mnist_for_manual_calculation":
            #optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
            loss_function = nn.CrossEntropyLoss()
        case _:
            raise ValueError(f"Unknown dataset_name. Can´t load optimizer and loss_function: {config.dataset_name}")
    return loss_function, optimizer


def load_config_File(config_path: str)->Config:
    return Config.from_yaml(config_path)

if __name__ == "__main__":
    config_path = "_Configuration/config.yaml"

    training_configurations = load_config_File(config_path)

    train_loader, test_loader = datasets_helper.get_dataloaders(training_configurations)
    start_NN(training_configurations, train_loader, test_loader)




