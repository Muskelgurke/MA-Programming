import torch.nn as nn
import torch
from _new.helpers.config_class import Config

def get_model(config: Config, sample_batch: tuple) -> nn.Module:
    """get model based on config and automatically set input and output sizes from sample batch
    Args:
        config: Config file
        sample_batch: tuple of (inputs, targets) from dataloader

    Returns:
        nn.Module: model
    """

    inputs, targets = sample_batch

    input_channels = inputs.shape[1]
    input_size = inputs.shape[2]
    num_classes = len(torch.unique(targets)) if targets.dim() == 1 else targets.shape[1]

    match config.dataset_name.lower():
        case "mnist" | "fashionmnist":
            input_size = 28 * 28
            output_size = 10

        case "cifar10":
            input_size = 32 * 32 * 3
            output_size = 10

        case "cifar100":
            input_size = 32 * 32 * 3
            output_size = 100

        case "cars":
            input_size = 224 * 224 * 3
            output_size = 196

        case "flowers":
            input_size = 224 * 224 * 3
            output_size = 102

        case "food":
            input_size = 224 * 224 * 3
            output_size = 101

        case "pets":
            input_size = 224 * 224 * 3
            output_size = 37

        case "yes_no":
            raise ValueError(f"Unknown dataset: {config.dataset_name}")

        case "speechCommands":
            raise ValueError(f"Unknown dataset: {config.dataset_name}")

        case "daliact":
            raise ValueError(f"Unknown dataset: {config.dataset_name}")

        case "cwru":
            raise ValueError(f"Unknown dataset: {config.dataset_name}")

        case _:
            raise ValueError(f"Unknown dataset: {config.dataset_name}")

    match config.model_type:
        case "AlexNet":
            hidden_size = 256
            return NN_fc(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        case "fc1024":
            hidden_size = 1024
            return NN_fc(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        case "fc4096":
            hidden_size = 4096
            return NN_fc(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        case "cnn":
            return CNN_for_CIFAR(output_size=output_size)

        case _:
            raise ValueError(f"Fully connected models not implemented for CIFAR-10")





