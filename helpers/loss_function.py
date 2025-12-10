import torch
from helpers.config_class import Config

def get_loss_function(config: Config) -> torch.nn.Module:
    """Returns a Pytorch Loss Function based on the Config-File settings."""
    loss_function = None

    match config.loss_function:
        case "cross_entropy":
            loss_function = torch.nn.CrossEntropyLoss()
        case "mse":
            loss_function = torch.nn.MSELoss()
        case _:
            raise ValueError(f"Unknown LOSS FUNCTION NAME. Can´t load loss function {config.loss_function}")

    return loss_function