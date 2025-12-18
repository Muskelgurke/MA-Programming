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
        print(f"Lauf {run_number}/{len(configs)}: ")
        print(f"Data: {config.dataset_name}, Model: {config.model_type}, Method:{config.training_method}")
        print(f"Config: LR={config.learning_rate}, Opt={config.optimizer}, Batch={config.batch_size}")
        print(f"Gerät: {device}")

        #try:

        start_nn_run(config_file=config,device=device,run_number=run_number,base_path=base_path)

        print(f"✅ Lauf {run_number} abgeschlossen.")

        #except Exception as e:
        #    print(f"❌ Fehler bei Lauf {run_number}: {e}")

    print(f"\n=== MULTI-RUN ABGESCHLOSSEN ===")

def get_device()->torch.device:
    """Gibt das Gerät zurück, auf dem das Training durchgeführt werden soll."""

    # Prüfe ob CUDA verfügbar ist (NVIDIA GPUs)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()

        # GPU Cluster mit 3 NVIDIA GPUs
        if gpu_count >= 3:
            device = torch.device("cuda:2")
            print(f"GPU Cluster erkannt: Nutze {torch.cuda.get_device_name(2)}")
            print(f"Verfügbare GPUs: {gpu_count}")
        # Einzelne NVIDIA GPU
        else:
            device = torch.device("cuda:0")
            print(f"Einzelne GPU erkannt: {torch.cuda.get_device_name(0)}")

    # Fallback auf CPU (z.B. Intel Arc Laptop)
    else:

        device = torch.device("cpu")
        print(f"Keine CUDA-GPU verfügbar. Nutze CPU.")
        exit()
        # Optional: Prüfe auf Intel Extension
        try:
            import intel_extension_for_pytorch as ipex
            print("Intel Extension für PyTorch gefunden (nicht aktiviert).")
        except ImportError:
            pass


    return device

if __name__ == "__main__":

    start_training(config_path = get_config(), device = get_device())