from typing import Any
import argparse
import torch
import sys
import datetime
import gc
import os

from helpers.config_class import MultiParamLoader, Config
from helpers.singlerun_manager_class import SingleRunManager
from pathlib import Path

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
    manager = SingleRunManager(config = config_file, device=device, run_number=run_number, base_path=base_path)
    try:
        manager.run()
    except Exception as e:
        print(f"Fehler im Lauf {run_number}: {e}")
        raise e
    finally:

        del manager
        clear_gpu_memory() # Räume GPU auf

def start_training(config_path: str):
    """
    Lädt die Multi-Parameter-Konfiguration und startet alle einzelnen Trainingsläufe.
    Überprüft dabei aber bereits erledigte Konfigurationen
    """
    print("=== MULTI-RUN KOORDINATOR STARTET ===")
    print(f"Lade Konfiguration von: {config_path}")
    config_loader = MultiParamLoader(config_path)
    config_loader.initialize()
    configs = config_loader.configs

    print(f"Starte insgesamt {len(configs)} Läufe...")

    # Check the following Paths for completed runs
    search_paths = [
        '/homes/sojagraf/25_1/MA-Programming/runs',
        '/homes/sojagraf/25_2/MA-Programming/runs',
        '/homes/sojagraf/25_3/MA-Programming/runs',
        # Auch den aktuellen Ordner scannen, falls wir im gleichen Ordner weitermachen
        'runs'
    ]

    #completed_configs_list = load_completed_configs(search_paths)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    base_path = f'runs/duc_{timestamp}'
    print(f"base_path: {base_path}")

    for i, config in enumerate(configs):
        run_number = i + 1

        is_done = False
        #for done_config in completed_configs_list:
        #    if is_same_experiment(config, done_config):
        #        is_done = True
        #        break
        print(f"\n{'=' * 40}")
        print(f"Prüfe Lauf {run_number}/{len(configs)}: {config.model_type} | {config.dataset_name} | {config.optimizer} | LR: {config.learning_rate}")
        if is_done:
            if is_done:
                print(f"⏩ ÜBERSPRUNGEN: Dieser Run wurde in einem der Scan-Verzeichnisse bereits gefunden.")
                continue

        print(f"\n{'=' * 40}")
        #device = torch.device(f"cuda:{config.cuda_device}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Lauf {run_number}/{len(configs)}: ")
        print(f"Data: {config.dataset_name}, Model: {config.model_type}, Method:{config.training_method}")
        print(f"Config: LR={config.learning_rate}, Opt={config.optimizer}, Batch={config.batch_size}")
        print(f"Gerät: {device}")
        start_nn_run(config_file=config,device=device,run_number=run_number,base_path=base_path)
        clear_gpu_memory()
        print(f"✅ Lauf {run_number} abgeschlossen.")

    print(f"\n=== MULTI-RUN ABGESCHLOSSEN ===")

def is_same_experiment(config_a: Config, config_b: Config) -> bool:
    """
    Vergleicht zwei Config-Objekte nur auf die Parameter, die das Training definieren.
    Ignoriert systemabhängige Dinge wie 'cuda_device' oder 'dataset_path'.
    """
    # Exakte Übereinstimmung von diesen Objekt-keys!!!!!
    relevant_keys = [
        'random_seed',
        'learning_rate',
        'training_method',
        'dataset_name',
        'batch_size',
        'model_type',
        'optimizer'
    ]

    dict_a = config_a.to_dict()
    dict_b = config_b.to_dict()

    for key in relevant_keys:
        val_a = dict_a.get(key)
        val_b = dict_b.get(key)

        # Fließkommazahlen-Vergleich (Lernrate etc.)
        if isinstance(val_a, float) and isinstance(val_b, float):
            if abs(val_a - val_b) > 1e-9:  # kleine Toleranz
                return False
        elif val_a != val_b:
            return False

    return True

def load_completed_configs(scan_directories: list[str]) -> list[Config]:
    """
    Durchsucht die angegebenen Verzeichnisse nach abgeschlossenen Trainings
    (erkannt durch Existenz von 'run_summary.yaml') und lädt deren Configs.
    """
    completed_configs = []
    print(f"Suche nach bereits fertigen Runs in: {scan_directories}")

    for directory in scan_directories:
        path = Path(directory)
        if not path.exists():
            print(f"Warnung: Verzeichnis existiert nicht: {directory}")
            continue

        # os.walk durchsuchte rekursif alle Unterordner
        for root, dirs, files in os.walk(path):
            # Kriterium: Ein Run ist fertig, wenn run_summary.yaml existiert
            if "run_summary.yaml" in files and "config.yaml" in files:
                config_path = Path(root) / "config.yaml"
                try:
                    # Config laden
                    loaded_conf = Config.from_yaml(str(config_path))
                    completed_configs.append(loaded_conf)
                except Exception as e:
                    print(f"Konnte Config nicht laden in {root}: {e}")

    print(f"Found {len(completed_configs)} completed runs.")
    return completed_configs

def create_config_list_of_completed_runs(directory: str) -> Any:
    """Überprüft die Konfigurationen auf Duplikate."""

    def search_file(directory=None, file=None):
        assert os.path.isdir(directory)
        completed_configs = []
        for cur_path, directories, files in os.walk(directory):
            if file in files:
                completed_configs.add(file)

        return None

def clear_gpu_memory():
    """Räumt den GPU-Speicher auf, falls CUDA verfügbar ist."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("GPU-Speicher wurde aufgeräumt.")

def get_device()->torch.device:
    """Gibt das Gerät zurück, auf dem das Training durchgeführt werden soll."""
    """
    # Prüfe ob CUDA verfügbar ist (NVIDIA GPUs)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device = torch.device(f"cuda:{config.cuda_device}")
        # GPU Cluster mit 3 NVIDIA GPUs
        if gpu_count >= 3:
            device = torch.device("cuda:1")
            print(f"GPU Cluster erkannt: Nutze {torch.cuda.get_device_name(2)}")
            print(f"Verfügbare GPUs: {gpu_count} - CudaDevice ID: {device}")

    return device"""
    pass

if __name__ == "__main__":
    start_training(config_path = get_config())