import torchvision.models as models


def print_resnet18_parameters():
    # Load pretrained ResNet18 model
    model = models.resnet18(pretrained=False)

    print("=" * 80)
    print("ResNet18 Model Parameters")
    print("=" * 80)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

        print(f"{name:50s} | Shape: {str(param.shape):30s} | Params: {num_params:>10,}")

    print("=" * 80)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    print("=" * 80)

    return model


# Call the function
if __name__ == "__main__":
    model = print_resnet18_parameters()
