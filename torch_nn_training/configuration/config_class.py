from dataclasses import dataclass, asdict
from typing import Optional, List
import yaml
import itertools

@dataclass
class Config:
    random_seed: int
    dataset_name: str
    learning_rate: float
    epoch_num: int
    batch_size: int
    dataset_path: str
    model_type: str
    training_method: str # e.g., "backpropagation", "forward_gradient"
    optimizer: str # e.g., "SGD", "Adam"
    loss_function: str # e.g., "CrossEntropy"
    momentum: float
    early_stopping_delta: float
    early_stopping: bool = False
    early_stopping_patience: Optional[int] = None
    augment_data: bool = False


    @classmethod
    def from_yaml(cls, yaml_path: str = "") -> "Config":
        """Load configuration from YAML file"""
        with open(yaml_path, "r") as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config from dictionary"""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert Config object to dictionary"""
        return asdict(self)

class MultiParamLoader:
    """Hilfsklasse zum Laden und Verarbeiten von Multi-Parameter-Konfigurationen"""

    @staticmethod
    def load_combined_config(yaml_path: str) -> tuple[dict, dict]:
        """Lade kombinierte Config mit base_config und multi_params"""
        with open(yaml_path, 'r') as file:
            full_config = yaml.safe_load(file)

        base_config = full_config.get('base_config', {})
        multi_params = full_config.get('multi_params', {})

        return base_config, multi_params

    @staticmethod
    def generate_combinations(multi_params: dict, base_config: Config) -> List[Config]:
        """Generiere alle Kombinationen aus Multi-Parameter-Config"""
        # Separate Parameter in Listen und Einzelwerte
        list_params = {}
        single_params = {}

        for key, value in multi_params.items():
            if isinstance(value, list) and len(value) > 1:
                list_params[key] = value
            else:
                # Einzelwerte oder Listen mit nur einem Element
                single_value = value[0] if isinstance(value, list) else value
                single_params[key] = single_value

        if not list_params:
            # Keine Listen gefunden, nur ein einzelnes Training
            config_dict = base_config.to_dict()
            config_dict.update(single_params)
            return [Config.from_dict(config_dict)]

        # Generiere alle Kombinationen der Listen-Parameter
        param_names = list(list_params.keys())
        param_values = list(list_params.values())
        combinations = list(itertools.product(*param_values))

        configs = []
        for combo in combinations:
            config_dict = base_config.to_dict()

            # Füge Einzelwerte hinzu
            config_dict.update(single_params)

            # Füge aktuelle Kombination hinzu
            for name, value in zip(param_names, combo):
                config_dict[name] = value

            configs.append(Config.from_dict(config_dict))

        return configs

    @staticmethod
    def get_combination_count(multi_params: dict) -> int:
        """Berechne die Anzahl der Kombinationen"""
        count = 1
        for value in multi_params.values():
            if isinstance(value, list):
                count *= len(value)
        return count

    @staticmethod
    def print_overview_of_config(multi_params: dict, base_config: dict = None):
        """Zeige eine Übersicht der Parameter-Kombinationen"""
        print("Multi-Parameter Übersicht:")
        print("-" * 40)

        if base_config:
            print("Base Config Parameter:")
            for key, value in base_config.items():
                if key not in multi_params:
                    print(f"  {key:18}: {value} (fest)")
            print()

        print("Variable Parameter:")
        for key, value in multi_params.items():
            if isinstance(value, list):
                if len(value) > 1:
                    print(f"  {key:18}: {len(value)} Werte -> {value}")
                else:
                    print(f"  {key:18}: {value[0]} (einzelner Wert)")
            else:
                print(f"  {key:18}: {value} (einzelner Wert)")

        total_combinations = MultiParamLoader.get_combination_count(multi_params)
        print(f"\nGesamte Kombinationen: {total_combinations}")
        print("-" * 40)
