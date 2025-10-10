import os
import sys
import argparse
from pathlib import Path
import datetime
import statistics
import time
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any

from configuration.config_class import Config, MultiParamLoader
from saver.saver_class import TorchModelSaver
from helpers.early_stopping import EarlyStopping
from trainer.trainer import Trainer
from tester.tester import Tester
import dataset.datasets as datasets_helper
import model.model as model_helper

def find_config_path()->str:
    """Findet den config.yaml Pfad mit mehreren Fallback-Strategien"""
    # 1. Command-line Argument (höchste Priorität)
    if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
        config_path = sys.argv[1]
        if os.path.exists(config_path):
            return config_path

    # 2. Umgebungsvariable
    env_config = os.getenv('CONFIG_PATH')
    if env_config and os.path.exists(env_config):
        return env_config

    # 3. Relativ zum Skript-Verzeichnis
    script_dir = Path(__file__).parent
    config_in_script_dir = script_dir / "config.yaml"
    if config_in_script_dir.exists():
        return str(config_in_script_dir)

    # 4. Relativ zum Projekt-Root (eine Ebene höher)
    project_root = script_dir.parent
    config_in_root = project_root / "config.yaml"
    if config_in_root.exists():
        return str(config_in_root)

    # 5. Aktuelles Arbeitsverzeichnis
    current_dir_config = Path.cwd() / "config.yaml"
    if current_dir_config.exists():
        return str(current_dir_config)

    # Fehler wenn nichts gefunden
    raise FileNotFoundError(
        f"config.yaml nicht gefunden. Gesucht in:\n"
        f"  - {config_in_script_dir}\n"
        f"  - {config_in_root}\n"
        f"  - {current_dir_config}\n"
        f"Verwende: python {sys.argv[0]} <config_path> oder setze CONFIG_PATH"
    )

def setup_argument_parser():
    """Erstellt ArgumentParser für Command-Line Interface"""
    parser = argparse.ArgumentParser(description='PyTorch Neural Network Training')
    parser.add_argument('--config', '-c', type=str,
                       help='Pfad zur config.yaml Datei')
    parser.add_argument('--mode', '-m', choices=['single', 'multi'],
                       default='multi',
                       help='Training Modus: single oder multi')
    parser.add_argument('--auto-confirm', '-y', action='store_true',
                       help='Automatisch bestätigen (für Server)')
    return parser


def run_multi_training(config_path: str = None,
                       auto_confirm: bool = False) -> None:
    """Führt Multi-Parameter-Training basierend auf kombinierter YAML-Konfiguration durch"""
    print("=== MULTI-PARAMETER TRAINING ===")
    config_path_is_none = config_path is None

    if config_path_is_none:
        config_path = find_config_path()

    print(f"Lade Konfiguration von: {config_path}")

    try:
        base_config_dict, multi_params = MultiParamLoader.load_combined_config(config_path)
        base_config = Config.from_dict(base_config_dict)

        # Übersicht anzeigen
        MultiParamLoader.print_overview_of_config(multi_params, base_config_dict)

        # Alle Kombinationen generieren
        configs = create_print_config_combinations(base_config, multi_params)

        if not auto_confirm:
            # Nur fragen wenn nicht auto-confirm
            user_input = input(f"Trainings starten? (Y/n): ").strip().lower()
            if user_input in ['n', 'no', 'nein']:
                print("Training abgebrochen.")
                return
            elif user_input not in ['y', 'yes', 'ja', '']:
                print("Ungültige Eingabe. Training abgebrochen.")
                return

        # Alle Kombinationen durchlaufen
        results, saver = run_all_combinations(configs)

        save_summary(results, saver)


    except FileNotFoundError as e:
        print(f"Konfigurationsdatei nicht gefunden: {e}")
        print("Stelle sicher, dass config.yaml existiert und base_config sowie multi_params enthält.")
    except Exception as e:
        print(f"Fehler beim Laden der Konfiguration: {e}")


def save_summary(results: list[Any], saver: TorchModelSaver):
    saver_for_summary = None
    for result in results:
        if 'error' not in result:
            saver_for_summary = saver
            break
    print_save_session_summary(results, saver_for_summary)


def run_all_combinations(configs: list[Config]) -> tuple[list[Any], TorchModelSaver]:
    results = []
    saver = None
    for i, config in enumerate(configs):
        print(f"\n{'=' * 60}")
        print(f"Training {i + 1}/{len(configs)}")
        print(f"Parameter: LR={config.learning_rate}, Method={config.training_method}, Seed={config.random_seed}")
        print(f"Batch Size: {config.batch_size}, Epochs: {config.epoch_num}")
        print(f"model_type: {config.model_type}, dataset_name: {config.dataset_name}")

        try:
            if torch.cuda.is_available() and torch.cuda.device_count() >= 3:
                device = torch.device("cuda:2")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            result, saver = start_nn_run(config=config,
                                         device=device,
                                         run_number=i + 1)

            results.append({
                'run': i + 1,
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
            print(f"Fehler beim Training {i + 1}: {e}")
            results.append({
                'run': i + 1,
                'config': {
                    'learning_rate': config.learning_rate,
                    'training_method': config.training_method,
                    'random_seed': config.random_seed
                },
                'error': str(e)
            })
    return results, saver


def create_print_config_combinations(base_config: Config, multi_params: dict) -> list[Config]:
    configs = MultiParamLoader.generate_combinations(multi_params, base_config)
    print(f"\nGenerierte {len(configs)} Konfigurationen:")

    for i, config in enumerate(configs[:5]):  # Zeige nur die ersten 5
        print(f"  {i + 1}. LR={config.learning_rate}, Method={config.training_method}, Seed={config.random_seed}")
    if len(configs) > 5:
        print(f"  ... und {len(configs) - 5} weitere")
    return configs


def create_config_for_combination(base_config: Config, param_names: list, combo: tuple) -> Config:
    """Erstellt eine neue Config mit den spezifischen Parameter-Werten"""
    config_dict = base_config.to_dict()

    # Parameter aus der Kombination überschreiben
    for name, value in zip(param_names, combo):
        config_dict[name] = value

    return Config.from_dict(config_dict)

def print_save_session_summary(results: list,
                               saver: TorchModelSaver) -> None:
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
    if saver is not None:
        saver.write_multi_session_summary(successful_runs=successful_runs, failed_runs=failed_runs)
    else:
        print("Warnung: kein saver verfügbar, um die Zusammenfassung zu speichern.")

def start_nn_run(config: Config,
                 device: torch.device,
                 run_number: int = 1)-> tuple[dict, TorchModelSaver]:
    """Startet einen einzelnen Trainingslauf basierend auf der gegebenen Konfiguration"""

    train_loader, test_loader = datasets_helper.get_dataloaders(config, device)

    model = model_helper.get_model(config)
    model.to(device)
    print("-" * 60)
    print(model)
    print("-" * 60)

    loss_function, optimizer = get_optimizer_and_lossfunction(config=config,
                                                              model=model)

    early_stopping = EarlyStopping(patience=config.early_stopping_patience,
                                   delta=config.early_stopping_delta) if config.early_stopping else None

    timestamp = datetime.datetime.now().strftime("%H_%M_%S")
    today = datetime.datetime.now().strftime("%Y_%m_%d")

    # Erweiterte Namensgebung für Multi-Parameter-Training
    run_path = f'runs/{today}/{timestamp}_run{run_number}_{config.dataset_name}_{config.model_type}_{config.training_method}_lr{config.learning_rate}_seed{config.random_seed}'
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
                      saver_class=saver,
                      early_stopping_class=early_stopping)

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

        if early_stopping.early_stop_nan_loss:
            run_time = time.time() - start_time


            saver.write_run_summary(config=config,
                                    total_training_time=run_time,
                                    train_acc=train_accs_per_epoch[-1],
                                    test_acc=test_accs_per_epoch[-1],
                                    test_loss=test_losses_per_epoch[-1],
                                    early_stop_info=early_stopping.early_stop_info)
            break


        early_stopping.check_validation(tester.avg_validation_loss, model)
        if early_stopping.early_stop:
            run_time = time.time() - start_time

            early_stop_info = {
                "reason": "early_stopping_patience",
                "stopped_at_epoch": epoch,
                "patience_reached": True
            }
            saver.write_run_summary(config=config,
                                    total_training_time=run_time,
                                    train_acc=train_accs_per_epoch[-1],
                                    test_acc=test_accs_per_epoch[-1],
                                    test_loss=test_losses_per_epoch[-1],
                                    early_stop_info=early_stop_info
                                    )
            print(f"Stopping early at Epoch {epoch + 1}")
            break


    print("-"* 60)
    print("Training and Testing completed")
    print(f"Durchschnittliche Zeit pro Epoch: {statistics.mean(epoch_times_per_epoch):.2f}s")
    print("-" * 60)

    saver.save_session(config= config,
                       model= model,
                       save_full_model= True)


    # Für Multi-Parameter-Training: Ergebnisse zurückgeben


    return {
        'final_train_acc': train_accs_per_epoch[-1] if train_accs_per_epoch else 0.0,
        'final_test_acc': test_accs_per_epoch[-1] if test_accs_per_epoch else 0.0,
        'final_train_loss': train_losses_per_epoch[-1] if train_losses_per_epoch else 0.0,
        'final_test_loss': test_losses_per_epoch[-1] if test_losses_per_epoch else 0.0,
        'avg_epoch_time': statistics.mean(epoch_times_per_epoch) if epoch_times_per_epoch else 0.0,
        'total_epochs': len(epoch_times_per_epoch)
    },saver

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

    parser = setup_argument_parser()
    args = parser.parse_args()

    try:
        if args.mode == "single":
            config_path = args.config or find_config_path()
            training_configurations = load_config_File(config_path)
            train_loader, test_loader = datasets_helper.get_dataloaders(training_configurations)
            start_nn_run(training_configurations, train_loader, test_loader)

        else:
            run_multi_training(args.config, args.auto_confirm)

    except FileNotFoundError as e:
        print(f"Fehler: {e}")
        sys.exit(1)
