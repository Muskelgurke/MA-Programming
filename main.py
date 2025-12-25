import torch
import sys
import datetime

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
    manager.run()

def start_training(config_path: str):
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
        device = torch.device(f"cuda:{config.cuda_device}")
        run_number = i + 1
        print(f"\n{'=' * 40}")
        print(f"Lauf {run_number}/{len(configs)}: ")
        print(f"Data: {config.dataset_name}, Model: {config.model_type}, Method:{config.training_method}")
        print(f"Config: LR={config.learning_rate}, Opt={config.optimizer}, Batch={config.batch_size}")
        print(f"Gerät: {device}")

        start_nn_run(config_file=config,device=device,run_number=run_number,base_path=base_path)
        clear_gpu_memory()
        print(f"✅ Lauf {run_number} abgeschlossen.")

    print(f"\n=== MULTI-RUN ABGESCHLOSSEN ===")

def clear_gpu_memory():
    """Räumt den GPU-Speicher auf, falls CUDA verfügbar ist."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("GPU-Speicher wurde aufgeräumt.")

def get_device()->torch.device:
    """Gibt das Gerät zurück, auf dem das Training durchgeführt werden soll."""

    # Prüfe ob CUDA verfügbar istw (NVIDIA GPUs)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device = torch.device(f"cuda:{config.cuda_device}")
        # GPU Cluster mit 3 NVIDIA GPUs
        if gpu_count >= 3:
            device = torch.device("cuda:1")
            print(f"GPU Cluster erkannt: Nutze {torch.cuda.get_device_name(2)}")
            print(f"Verfügbare GPUs: {gpu_count} - CudaDevice ID: {device}")

    return device

if __name__ == "__main__":

    start_training(config_path = get_config())