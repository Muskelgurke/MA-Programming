import torch
from helpers.config_class import Config

def get_optimizer (config: Config, model: torch.nn.Module)-> torch.optim.Optimizer:
    """"Returns a Pytorch Optimizer based on the Config-File settings."""
    optimizer = None

    match config.optimizer:
        case "sgd":
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=config.learning_rate,
                                        momentum=config.momentum)
        case "adam":
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config.learning_rate)

        case _:
            raise ValueError(f"Unknown OPTIMIZER NAME. Can´t load optimizer {config.dataset_name}")

    return optimizer