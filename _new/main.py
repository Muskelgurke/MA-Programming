import torch
import pytest
import sys

from helpers.config_class import MultiParamLoader, Config

from helpers.singlerun_manager_class import SingleRunManager
from pathlib import Path
import datetime

def get_config()->str:
    """Gibt den config.yaml Pfad zurück"""
    script_dir = Path(__file__).parent
    config_in_script_dir = script_dir / "config.yaml" # change here quickly the config
    if config_in_script_dir.exists():
        return str(config_in_script_dir)
    else:
        print("'config.yaml' not found in script directory. Please provide the correct path.")
        sys.exit(1)



def start_nn_run(config_file: Config, device: torch.device, run_number: int, base_path= str) -> None:
    print(f"Start {run_number} Run")
    manager = SingleRunManager(config = config_file, device=device, run_number=run_number, base_path=base_path)
    manager.run()


def start_training(config_path: str, device: torch.device):
    """
    Lädt die Multi-Parameter-Konfiguration und startet alle einzelnen Trainingsläufe.
    """
    print("=== MULTI-RUN KOORDINATOR STARTET ===")
    print(f"Lade Konfiguration von: {config_path}")
    config_loader = MultiParamLoader(config_path)
    config_loader.initialize()
    configs = config_loader.configs

    print(f"Starte insgesamt {len(configs)} Läufe...")

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    base_path = f'runs/{timestamp}'
    print(f"base_path: {base_path}")

    for i, config in enumerate(configs):
        run_number = i + 1
        print(f"\n{'=' * 40}")
        print(f"Lauf {run_number}/{len(configs)}: Starte Experiment")
        print(f"Config: LR={config.learning_rate}, Method={config.training_method}")
        print(f"Gerät: {device}")

        try:

            start_nn_run(config_file=config,
                         device=device,
                         run_number=run_number,
                         base_path=base_path)

            print(f"✅ Lauf {run_number} abgeschlossen.")

        except Exception as e:
            print(f"❌ Fehler bei Lauf {run_number}: {e}")

    print(f"\n=== MULTI-RUN ABGESCHLOSSEN ===")

def get_device()->torch.device:
    """Gibt das Gerät zurück, auf dem das Training durchgeführt werden soll."""
    if torch.cuda.is_available() and torch.cuda.device_count() >= 3:
        device = torch.device("cuda:2")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

if __name__ == "__main__":

    start_training(config_path = get_config(), device = get_device())