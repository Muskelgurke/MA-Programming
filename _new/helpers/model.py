import torch.nn as nn
import torch
import torchvision.models as models
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

    specs = {
        "input_channels": input_channels,
        "input_size": input_size,
        "num_classes": num_classes
    }

    match config.model_type.lower():
        case "alexnet":
            return get_adapted_model(models.alexnet ,specs)
        case "vgg16":
            return get_adapted_model(models.vgg16 ,specs)
        case "resnet18":
            return get_adapted_model(models.resnet18 ,specs)
        case "resnet34":
            return get_adapted_model(models.resnet34 ,specs)
        case "resnet50":
            return get_adapted_model(models.resnet50 ,specs)
        case "mobilenet_v2":
            return get_adapted_model(models.mobilenet_v2 ,specs)
        case "densenet":
            return get_adapted_model(models.densenet121 ,specs)
        case "efficientnet_v2_s":
            return get_adapted_model(models.efficientnet_v2_s ,specs)
        case _:
            raise ValueError(f"Unknown model type: {config.model_type}")


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


def get_adapted_model(model_fn= type[nn.Module], specs= dict):
    model = model_fn(weights=None)
    num_classes = specs.get("num_classes")
    input_channels = specs.get("input_channels")
    if num_classes is None:
        raise ValueError("specs must contain 'num_classes'")
    if input_channels is None:
        raise ValueError("specs must contain 'input_channels'")

    input_channels_are_not_3 = input_channels !=3 # for MNIST and fashionMNIST

    if input_channels_are_not_3:
        if hasattr(model, "classifier"):
            first_layer = model.features[0]
            # handle Conv2dNormActivation wrapper (efficientNet)
            if hasattr(first_layer, '0'):
                first_conv = first_layer[0]
            else:
                first_conv = first_layer

            new_conv = nn.Conv2d(in_channels=input_channels,
                                          out_channels= first_conv.out_channels,
                                          kernel_size= first_conv.kernel_size,
                                          stride= first_conv.stride,
                                          padding= first_conv.padding,
                                          bias=first_conv.bias is not None)

            if hasattr(first_layer, '0'):
                first_layer[0] = new_conv
            else:
                model.features[0] = new_conv


        elif hasattr(model, "conv1"):
            # For ResNet Models
            model.conv1 = nn.Conv2d(in_channels= input_channels,
                                    out_channels= model.conv1.out_channels,
                                    kernel_size= model.conv1.kernel_size,
                                    stride= model.conv1.stride,
                                    padding= model.conv1.padding,
                                    bias= False
            )

    # 3. Output-Layer anpassen
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            # AlexNet, VGG
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            # MobileNet, DenseNet, EfficientNet
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        # ResNet-Modelle
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model




