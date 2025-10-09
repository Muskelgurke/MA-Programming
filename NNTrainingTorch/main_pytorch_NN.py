import datetime
import statistics
import time
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from NNTrainingTorch.helpers.training_metrics_class import TrainingMetrics
from NNTrainingTorch.helpers.config_class import Config, MultiParamLoader
from NNTrainingTorch.helpers.saver_class import TorchModelSaver
from NNTrainingTorch.helpers.results_of_epochs import results_of_epochs
from NNTrainingTorch.helpers.early_stopping import EarlyStopping
from NNTrainingTorch.helpers.trainer import Trainer
from NNTrainingTorch.helpers.tester import Tester
import NNTrainingTorch.helpers.datasets as datasets_helper
import NNTrainingTorch.helpers.model as model_helper

def run_multi_training():
    """Führt Multi-Parameter-Training basierend auf kombinierter YAML-Konfiguration durch"""
    print("=== MULTI-PARAMETER TRAINING ===")
    # Kombinierte Config laden
    config_path = "_Configuration/config.yaml"

    try:
        base_config_dict, multi_params = MultiParamLoader.load_combined_config(config_path)
        base_config = Config.from_dict(base_config_dict)

        # Übersicht anzeigen
        MultiParamLoader.print_summary(multi_params, base_config_dict)

        # Alle Kombinationen generieren
        configs = MultiParamLoader.generate_combinations(multi_params, base_config)

        print(f"\nGenerierte {len(configs)} Konfigurationen:")
        for i, config in enumerate(configs[:5]):  # Zeige nur die ersten 5
            print(f"  {i+1}. LR={config.learning_rate}, Method={config.training_method}, Seed={config.random_seed}")
        if len(configs) > 5:
            print(f"  ... und {len(configs) - 5} weitere")

        # Benutzer-Bestätigung
        user_input = input(f"\nMöchtest du {len(configs)} Trainings starten? (y/n): ")
        if user_input.lower() != 'y':
            print("Training abgebrochen.")
            return

        # Alle Kombinationen durchlaufen
        results = []
        for i, config in enumerate(configs):
            print(f"\n{'='*60}")
            print(f"Training {i+1}/{len(configs)}")
            print(f"Parameter: LR={config.learning_rate}, Method={config.training_method}, Seed={config.random_seed}")
            print(f"{'='*60}")

            try:
                train_loader, test_loader = datasets_helper.get_dataloaders(config)
                result = start_NN(config, train_loader, test_loader, run_number=i+1)

                results.append({
                    'run': i+1,
                    'config': {
                        'learning_rate': config.learning_rate,
                        'training_method': config.training_method,
                        'random_seed': config.random_seed,
                        'batch_size': config.batch_size,
                        'epoch_num': config.epoch_num
                    },
                    'final_train_acc': result.get('final_train_acc', 0),
                    'final_test_acc': result.get('final_test_acc', 0),
                    'final_train_loss': result.get('final_train_loss', 0),
                    'final_test_loss': result.get('final_test_loss', 0)
                })

            except Exception as e:
                print(f"Fehler beim Training {i+1}: {e}")
                results.append({
                    'run': i+1,
                    'config': {
                        'learning_rate': config.learning_rate,
                        'training_method': config.training_method,
                        'random_seed': config.random_seed
                    },
                    'error': str(e)
                })

        print_training_summary(results)

    except FileNotFoundError as e:
        print(f"Konfigurationsdatei nicht gefunden: {e}")
        print("Stelle sicher, dass config.yaml existiert und base_config sowie multi_params enthält.")
    except Exception as e:
        print(f"Fehler beim Laden der Konfiguration: {e}")

def create_config_for_combination(base_config: Config, param_names: list, combo: tuple) -> Config:
    """Erstellt eine neue Config mit den spezifischen Parameter-Werten"""
    config_dict = base_config.to_dict()

    # Parameter aus der Kombination überschreiben
    for name, value in zip(param_names, combo):
        config_dict[name] = value

    return Config.from_dict(config_dict)

def print_training_summary(results: list):
    """Druckt eine Zusammenfassung aller Trainings-Ergebnisse"""
    successful_runs = [r for r in results if 'error' not in r]
    failed_runs = [r for r in results if 'error' in r]

    if successful_runs:
        # Nach Test-Accuracy sortieren
        successful_runs.sort(key=lambda x: x['final_test_acc'], reverse=True)

        print(f"\nERFOLGREICHE LÄUFE ({len(successful_runs)}):")
        print("-" * 80)
        print(f"{'Rang':<4}{'Run'} {'LR':<8} {'Method':<6} {'Seed':<4} {'Train Acc':<10} {'Test Acc':<9} {'Train Loss':<11} {'Test Loss':<10}")
        print("-" * 80)

        for rank, result in enumerate(successful_runs, 1):
            config = result['config']
            print(f"{rank:<4} {result['run']} {config['learning_rate']:<8} {config['training_method']:<6} {config['random_seed']:<4} "
                  f"{result['final_train_acc']:<10.2f} {result['final_test_acc']:<9.2f} "
                  f"{result['final_train_loss']:<11.4f} {result['final_test_loss']:<10.4f}")

        # Bestes Ergebnis hervorheben
        best = successful_runs[0]
        print(f"\nBESTES ERGEBNIS:")

        print(f"   Test Accuracy: {best['final_test_acc']:.2f}%")
        print(f"   Learning Rate: {best['config']['learning_rate']}")
        print(f"   Training Method: {best['config']['training_method']}")
        print(f"   Random Seed: {best['config']['random_seed']}")

    if failed_runs:
        print(f"\nFEHLGESCHLAGENE LÄUFE ({len(failed_runs)}):")
        print("-" * 50)
        for result in failed_runs:
            print(f"Lauf {result['run']}: {result['error']}")

    print("="*80)

def start_NN(config: Config, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, run_number: int = 1):
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
    # Erweiterte Namensgebung für Multi-Parameter-Training
    run_path = f'runs/{timestamp}_run{run_number}_{config.dataset_name}_{config.model_type}_{config.training_method}_lr{config.learning_rate}_seed{config.random_seed}'
    tensorboard_writer = SummaryWriter(log_dir=run_path)
    saver = TorchModelSaver(tensorboard_writer.log_dir) # eigener Saver gebaut. Plotting etc.

    trainer = Trainer(config_file=config,
                      model=model,
                      data_loader=train_loader,
                      loss_function=loss_function,
                      optimizer=optimizer,
                      device=device,
                      total_epochs=config.epoch_num,
                      seed=config.random_seed,
                      tensorboard_writer=tensorboard_writer,
                      saver_class=saver)

    tester = Tester(config_file=config,
                    model=model,
                    test_loader=test_loader,
                    loss_function=loss_function,
                    device=device,
                    total_epochs=config.epoch_num,
                    seed=config.random_seed,
                    tensorboard_writer=tensorboard_writer)

    train_losses_per_epoch = []
    train_accs_per_epoch = []
    test_losses_per_epoch = []
    test_accs_per_epoch = []
    epoch_times_per_epoch = []

    # To Do
    for epoch in range(config.epoch_num):
        start_time = time.time()

        training_results = trainer.train_epoch(epoch_num=epoch)

        tester_results = tester.validate_epoch(epoch_num=epoch)

        epoch_time = time.time() - start_time
        epoch_times_per_epoch.append(epoch_time)

        saver.write_epoch_metrics(training_metrics=training_results,
                                  test_loss=tester_results.test_loss_per_epoch,
                                  test_accuracy=tester_results.test_acc_per_epoch,
                                  epoch=epoch)

        train_losses_per_epoch.append(training_results.epoch_avg_train_loss)
        train_accs_per_epoch.append(training_results.epoch_train_acc)

        test_losses_per_epoch.append(tester_results.test_loss_per_epoch)
        test_accs_per_epoch.append(tester_results.test_acc_per_epoch)

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar('Train/Loss - Epoch',
                              training_results.epoch_avg_train_loss,
                              epoch)
            tensorboard_writer.add_scalar('Train/Accuracy - Epoch',
                              training_results.epoch_train_acc)
            tensorboard_writer.add_scalar('Test/Loss - Epoch',
                              tester_results.test_loss_per_epoch,
                              epoch)
            tensorboard_writer.add_scalar('Test/Accuracy - Epoch',
                              tester_results.test_acc_per_epoch,
                              epoch)

        early_stopping(tester.avg_validation_loss, model)
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
        epoch_times =    epoch_times_per_epoch
    )

    saver.save_session(results_of_epoch=results,
                       config= config,
                       model= model,
                       save_full_model=True)


    ###########################################################################################
    # Linear Regresssion Demo Extra Output
    if config.dataset_name == "demo_linear_regression":
        print(f'\nYour New Funktion: y = {model.linear.weight.item():.6f} * X + {model.linear.bias.item():.6f}\n')
    ###########################################################################################

    # Für Multi-Parameter-Training: Ergebnisse zurückgeben
    return {
        'final_train_acc': train_accs_per_epoch[-1] if train_accs_per_epoch else 0.0,
        'final_test_acc': test_accs_per_epoch[-1] if test_accs_per_epoch else 0.0,
        'final_train_loss': train_losses_per_epoch[-1] if train_losses_per_epoch else 0.0,
        'final_test_loss': test_losses_per_epoch[-1] if test_losses_per_epoch else 0.0,
        'avg_epoch_time': statistics.mean(epoch_times_per_epoch) if epoch_times_per_epoch else 0.0,
        'total_epochs': len(epoch_times_per_epoch)
    }

def get_optimizer_and_lossfunction(config: Config, model: torch.nn.Module) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    loss_function = None
    optimizer = None
    match config.dataset_name:
        case "demo_linear_regression":
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
            loss_function = nn.MSELoss()
        case "mnist" | "fashionmnist" | "small_mnist_for_manual_calculation":
            if config.optimizer == "sgd":
                optimizer = torch.optim.SGD(model.parameters(),
                                            lr=config.learning_rate,
                                            momentum=config.momentum)
            if config.optimizer == "adam":
                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=config.learning_rate)

            loss_function = nn.CrossEntropyLoss()
        case _:
            raise ValueError(f"Unknown dataset_name. Can´t load optimizer and loss_function: {config.dataset_name}")


    return loss_function, optimizer


def load_config_File(config_path: str)->Config:
    """Lade Config für Single-Training aus der kombinierten YAML-Datei"""
    base_config_dict, _ = MultiParamLoader.load_combined_config(config_path)
    return Config.from_dict(base_config_dict)

if __name__ == "__main__":
    print("=== PyTorch Neural Network Training ===")
    print("1. Einzelnes Training (normale Config)")
    print("2. Multi-Parameter Training (mehrere Kombinationen)")

    choice = input("Wähle eine Option (1 oder 2): ")

    if choice == "1":
        # Normales einzelnes Training
        config_path = "_Configuration/config.yaml"
        training_configurations = load_config_File(config_path)
        train_loader, test_loader = datasets_helper.get_dataloaders(training_configurations)
        start_NN(training_configurations, train_loader, test_loader)

    elif choice == "2":
        # Multi-Parameter Training
        run_multi_training()

    else:
        print("Ungültige Auswahl. Starte normales Training...")
        config_path = "_Configuration/config.yaml"
        training_configurations = load_config_File(config_path)
        train_loader, test_loader = datasets_helper.get_dataloaders(training_configurations)
        start_NN(training_configurations, train_loader, test_loader)
