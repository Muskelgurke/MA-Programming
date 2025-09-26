from dataclasses import dataclass, asdict
from typing import Optional
import yaml

@dataclass
class Config:
    random_seed: int
    dataset_name: str
    learning_rate: float
    num_epochs: int
    batch_size: int
    dataset_path: str
    model_type: str
    training_method: str # e.g., "backpropagation", "forward_gradient"
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
