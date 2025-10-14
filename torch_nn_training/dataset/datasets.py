import torch.utils.data
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from configuration.config_class import Config


def get_dataloaders(config: Config, device: torch.device) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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
            train_loader, test_loader = get_mnist_dataloaders(config, device)
            #if device == torch.device("cuda"):
             #   train_loader, test_loader = _push_all_data_to_gpu(test_loader=test_loader,
              #                                                    train_loader=train_loader,
               #                                                   config=config,
                #                                                  device=device)
        case "fashionmnist":
            train_loader, test_loader = get_fashion_mnist_dataloaders(config, device)

        case "cifar10":
            train_loader, test_loader = get_cifar10_dataloaders(config,device)

        case "cifar100":
            train_loader, test_loader = get_cifar100_dataloaders(config,device)

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

def get_mnist_dataloaders(config: Config, device: torch.device)-> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load MNIST dataset and return training and test dataloaders.

    Args:
        config (Config): Configuration object containing dataset path and batch size.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test dataloaders.
    """
    # Mittelwert und Standardabweichung für MNIST
    # mean = 0.1307
    # std = 0.3081
    pin_memory = False
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
                                               num_workers=0,
                                               pin_memory=pin_memory,
                                               drop_last=True
                                               )

    # Load test dataset der 10.000 Bilder hat
    test_dataset = datasets.MNIST(root=config.dataset_path,
                                  train=False,
                                  download=True,
                                  transform=transform_mnist)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=pin_memory,
                                              drop_last=False
                                              )

    return train_loader, test_loader
def get_fashion_mnist_dataloaders(config: Config, device: torch.device)-> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load Fashion-MNIST dataset and return training and test dataloaders.

    Args:
        config (Config): Configuration object containing dataset path and batch size.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test dataloaders.
    """
    # Mittelwert und Standardabweichung für MNIST
    mean = 0.2860
    std = 0.3530

    transform_fashion = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))])

    train_dataset = datasets.FashionMNIST(root=config.dataset_path,
                                   train=True,
                                   download=True,
                                   transform=transform_fashion)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True if device.type == 'cuda' else False,
                                               drop_last=True
                                               )


    test_dataset = datasets.FashionMNIST(root=config.dataset_path,
                                  train=False,
                                  download=True,
                                  transform=transform_fashion)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True if device.type == 'cuda' else False,
                                              drop_last=False
                                              )

    return train_loader, test_loader


def get_cifar10_dataloaders(config: Config, device: torch.device) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load CIFAR-10 dataset and return training and test dataloaders.

    Args:
        config (Config): Configuration object containing dataset path and batch size.
        device (torch.device): Device to determine pin_memory setting.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test dataloaders.
    """
    # Mittelwert und Standardabweichung für CIFAR-10 (RGB-Kanäle)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    if config.augment_data:
        transform_cifar10 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform_cifar10 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.CIFAR10(root=config.dataset_path,
                                     train=True,
                                     download=True,
                                     transform=transform_cifar10)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True if device.type == 'cuda' else False,
                                               drop_last=True)

    test_dataset = datasets.CIFAR10(root=config.dataset_path,
                                    train=False,
                                    download=True,
                                    transform=transform_test)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True if device.type == 'cuda' else False,
                                              drop_last=False)

    return train_loader, test_loader

def get_cifar100_dataloaders(config: Config, device: torch.device) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    return train_loader, test_loader

def _push_all_data_to_gpu(test_loader: torch.utils.data.DataLoader,
                          train_loader: torch.utils.data.DataLoader,
                          config: Config,
                          device: torch.device) -> torch.utils.data.DataLoader:
    """
    Push all data from the dataloader to GPU memory.

    Args:
        dataloader (DataLoader): DataLoader containing the dataset.
        device (torch.device): Device to push the data to.

    Returns:
        DataLoader: DataLoader with all data pushed to GPU memory.
    """
    def get_dataset(dataloader:DataLoader) -> torch.utils.data.Tensordataset:
        stack_x = list()
        stack_y = list()
        for idx, (x,y) in enumerate(dataloader):
            stack_x.append(x)
            stack_y.append(y)
        x = torch.squeeze(torch.stack(stack_x),1).to(device)
        y = torch.squeeze(torch.stack(stack_y),1).to(device)
        dataset = torch.utils.data.TensorDataset(x,y)
        return dataset

    train_dataset = get_dataset(train_loader)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               pin_memory=False,
                                               drop_last=True,
                                               num_workers=0)
    test_dataset = get_dataset(test_loader)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=config.batch_size,
                                                pin_memory=False,
                                                drop_last=False,
                                                num_workers=0)

    return train_loader, test_loader