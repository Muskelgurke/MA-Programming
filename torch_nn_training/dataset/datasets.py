import torch.utils.data
from torchvision import datasets, transforms
from configuration.config_class import Config


def get_dataloaders(config: Config, device=None):
    """
    Load dataset and return training and test dataloaders.

    Args:
        config (Config): Configuration object containing dataset and batch size.
        device (torch.device): Device to use.
    Returns:
        Tuple[DataLoader, DataLoader]: Training and test dataloaders.
        When device is 'cuda', the entire dataset is loaded to GPU memory.
    """
    train_loader = None
    test_loader = None

    match config.dataset_name.lower():
        case "mnist":
            train_loader, test_loader = get_mnist_dataloaders(config)
        case "demo_linear_regression":
            train_loader, test_loader = get_linear_regression_dataloaders(config)
        case _:
            raise ValueError(f"Unknown dataset-name: {config.dataset_name}")

    return train_loader, test_loader

def get_linear_regression_dataloaders(config: Config) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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
    train_dataset = torch.utils.data.TensorDataset(x_data, y_data)
    test_dataset = torch.utils.data.TensorDataset(x_data, y_data)
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, test_loader

def get_mnist_dataloaders(config: Config)-> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load MNIST dataset and return training and test dataloaders.

    Args:
        config (Config): Configuration object containing dataset path and batch size.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test dataloaders.
    """
    num_workers = min(8, torch.get_num_threads())
    # Mittelwert und Standardabweichung für MNIST
    # mean = 0.1307
    # std = 0.3081
    transform_mnist = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    # Load training dataset
    train_dataset = datasets.MNIST(root=config.dataset_path,
                                   train=True,
                                   download=True,
                                   transform=transform_mnist)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               persistent_workers=True if num_workers > 0 else False,
                                               prefetch_factor=2 if num_workers > 0 else None
                                               )

    # Load test dataset der 10.000 Bilder hat
    test_dataset = datasets.MNIST(root=config.dataset_path,
                                  train=False,
                                  download=True,
                                  transform=transform_mnist)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              persistent_workers=True if num_workers > 0 else False,
                                              )

    return train_loader, test_loader


def get_cifar10_dataloaders(config: Config):
    num_workers = min(8, torch.get_num_threads())

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root=config.dataset_path,
                                     train=True,
                                     download=True,
                                     transform=transform_train)

    test_dataset = datasets.CIFAR10(root=config.dataset_path,
                                    train=False,
                                    download=True,
                                    transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               persistent_workers=True if num_workers > 0 else False,
                                               prefetch_factor=2
                                               )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              persistent_workers=True if num_workers > 0 else False
                                              )

    return train_loader, test_loader

def load_dataloader_to_gpu(dataloader, device):
    """Load entire dataloader to GPU memory"""
    gpu_data = []
    gpu_targets = []

    for data, targets in dataloader:
        gpu_data.append(data.to(device, non_blocking=True))
        gpu_targets.append(targets.to(device, non_blocking=True))

    # Concatenate all batches
    all_data = torch.cat(gpu_data, dim=0)
    all_targets = torch.cat(gpu_targets, dim=0)

    # Create new dataset and dataloader
    gpu_dataset = torch.utils.data.TensorDataset(all_data, all_targets)
    return torch.utils.data.DataLoader(gpu_dataset,
                                     batch_size=dataloader.batch_size,
                                     shuffle=True)