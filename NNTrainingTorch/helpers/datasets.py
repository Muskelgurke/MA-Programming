import torch.utils.data
from torchvision import datasets, transforms
from NNTrainingTorch.helpers.config_class import Config

def get_linear_regression_dataloaders(config: Config) -> torch.utils.data.DataLoader:
    """
    Load a simple linear regression dataset and return training and test dataloaders.

    Args:
        config (Config): Configuration object containing dataset path and batch size.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test dataloaders.
    """
    x_data = torch.tensor([[1.0], [2.0], [-1.0], [1.0]], dtype=torch.float32)  # Shape: (4, 1)
    y_data = torch.tensor([[2.0], [3.0], [-1.0], [1.0]], dtype=torch.float32)  # Shape: (4, 1)


    dataset = torch.utils.data.TensorDataset(x_data, y_data)

    train_size = len(dataset)
    test_size = 0
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, None

def get_mnist_dataloaders(config: Config)-> torch.utils.data.DataLoader:
    """
    Load MNIST dataset and return training and test dataloaders.

    Args:
        config (Config): Configuration object containing dataset path and batch size.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test dataloaders.
    """
    # ToDO: Augmentation?
    # Mittelwert und Standardabweichung für MNIST
    # mean = 0.1307
    # std = 0.3081
    transform_mnist = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    # Load training dataset
    train_dataset = datasets.MNIST(config.dataset_path, train=True, download=True, transform=transform_mnist)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Load test dataset der 10.000 Bilder hat
    test_dataset = datasets.MNIST(config.dataset_path, train=False, download=True, transform=transform_mnist)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader

def get_dataloaders(config: Config):
    """
    Load dataset and return training and test dataloaders.

    Args:
        config (Config): Configuration object containing dataset and batch size.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test dataloaders.
    """
    match config.dataset_name.lower():
        case "mnist" | "fashionmnist":
            return get_mnist_dataloaders(config)

        case "linear_regression":
            return get_linear_regression_dataloaders(config)

        case _:
            raise ValueError(f"Unknown dataset: {config.dataset_name}")
