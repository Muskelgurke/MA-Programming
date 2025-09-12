from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class Config:
    dataset: str
    randomSeed: int
    learningRate: float
    numEpochs: int
    batchSize: int
    path2save: Optional[str] = None

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