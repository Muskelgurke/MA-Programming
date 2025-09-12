import yaml
from NNTrainingTorch.helpers.Config import Config

def load_config_File(config_path: str)->Config:
    config = Config.from_yaml(config_path)
    return config

if __name__ == "__main__":
    config_path = "_Configuration/config.yaml"
    training_configurations = load_config_File(config_path)



