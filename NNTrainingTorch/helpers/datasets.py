import torch.utils.data
from torchvision import datasets, transforms
from NNTrainingTorch.helpers.config_class import Config

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

    # Select dataset class based on configuration
    if config.dataset_name.lower() == "mnist":
        return get_mnist_dataloaders(config)
    #elif config.dataset.lower() == "fashionmnist":

    #elif config.dataset.lower() == "cifar10":

    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")

    # Load training dataset
    train_dataset = DatasetClass(config.dataset_path, train=True, download=True, transform=flatten_and_cast)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **dataloader_kwargs)

    # Load test dataset
    test_dataset = DatasetClass('../_Dataset/', train=False, download=True, transform=flatten_and_cast)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **dataloader_kwargs)

    return train_loader, test_loader