import torch.utils.data
from torchvision import datasets, transforms
from torchaudio import datasets as audio_datasets
from torchaudio import transforms as audio_transforms
from _new.helpers.config_class import Config

def get_dataloaders(config: Config, device: torch.device) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load dataset and return TRAIN and TEST dataloaders.

    Args:
        config (Config): Configuration object containing dataset and batch size.
        device (torch.device): Device to use.
    Returns:
        Tuple[DataLoader, DataLoader]: Training and Test dataloaders.
        When device is 'cuda', the entire dataset is loaded to GPU memory.
    """

    train_loader = None
    test_loader = None

    match config.dataset_name.lower():
        case "mnist":
            train_loader, test_loader = get_mnist_dataloaders(config, device)

        case "fashionmnist":
            train_loader, test_loader = get_fashion_mnist_dataloaders(config, device)

        case "cifar10":
            train_loader, test_loader = get_cifar10_dataloaders(config,device)

        case "cifar100":
            train_loader, test_loader = get_cifar100_dataloaders(config,device)

        case "StanfordCars":
            train_loader, test_loader = get_standCars_dataloaders(config,device)

        case "Flower102":
            train_loader, test_loader = get_flower102_dataloaders(config, device)

        case "Food101":
            train_loader, test_loader = get_food101_dataloaders(config, device)

        case "OxfordIIITPet":
            train_loader, test_loader = get_oxfordPet_dataloaders(config,device)

        case "yes_no":
            # TorchAudio

        case "SpeechCommands":
            # TorchAudio - Warden
        case "DaLiAc":
            # https://www.mad.tf.fau.de/research/activitynet/daliac-daily-life-activities/
            # LOW Prio -> wäre cool
            # wichtiger als cwru

        case "cwru":
            # MaschineDataset
            # LOW Prio -> wäre cool



        case "demo_linear_regression":
            train_loader, test_loader = get_linear_regression_dataloaders(config)

        case _:
            raise ValueError(f"Unknown dataset-name: {config.dataset_name}")

    return train_loader, test_loader

def _create_dataloaders(train_dataset: torch.utils.data.Dataset,test_dataset: torch.utils.data.Dataset,
                        config: Config,
                        device: torch.device) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """ Private helper function to create dataloaders from datasets. """
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True if device.type == 'cuda' else False,
                                               drop_last=True
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True if device.type == 'cuda' else False,
                                              drop_last=False
                                              )
    return train_loader, test_loader

def get_oxfordPet_dataloaders(config: Config, device: torch.device)-> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
        Load OxfordIIITPet dataset and return training and test dataloaders.

        Args:
            config (Config): Configuration object containing dataset path and batch size.

        Returns:
            Tuple[DataLoader, DataLoader]: Training and test dataloaders.

        """
    transform_pet = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])

    train_dataset = datasets.OxfordIIITPet(root=config.dataset_path,split='trainval',download=True,transform=transform_pet)

    test_dataset = datasets.OxfordIIITPet(root=config.dataset_path,split='test',download=True,transform=transform_pet)

    return _create_dataloaders(train_dataset, test_dataset, config, device)

def get_food101_dataloaders(config: Config, device: torch.device)-> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
        Load Food101 dataset and return training and test dataloaders.

        Args:
            config (Config): Configuration object containing dataset path and batch size.

        Returns:
            Tuple[DataLoader, DataLoader]: Training and test dataloaders.

        """
    transform_food = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])

    train_dataset = datasets.Food101(root=config.dataset_path,split='train',download=True,transform=transform_food)

    test_dataset = datasets.Food101(root=config.dataset_path,split='test',download=True,transform=transform_food)

    return _create_dataloaders(train_dataset, test_dataset, config, device)

def get_flower102_dataloaders(config: Config, device: torch.device)-> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
        Load Flower102 dataset and return training and test dataloaders.

        Args:
            config (Config): Configuration object containing dataset path and batch size.

        Returns:
            Tuple[DataLoader, DataLoader]: Training and test dataloaders.
        """
    transform_flowers = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))])

    train_dataset = datasets.Flowers102(root=config.dataset_path,split='train',download=True,transform=transform_flowers)

    test_dataset = datasets.Flowers102(root=config.dataset_path,split='test',download=True,transform=transform_flowers)

    return _create_dataloaders(train_dataset, test_dataset, config, device)

def get_standCars_dataloaders(config: Config, device: torch.device)-> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
        Load Standford Cars dataset and return training and test dataloaders.

        Args:
            config (Config): Configuration object containing dataset path and batch size.

        Returns:
            Tuple[DataLoader, DataLoader]: Training and test dataloaders.
        """

    transform_cars = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    train_dataset = datasets.StanfordCars(root=config.dataset_path,split='train',download=True,transform=transform_cars)

    test_dataset = datasets.StanfordCars(root=config.dataset_path,split='test',download=True,transform=transform_cars)

    return _create_dataloaders(train_dataset, test_dataset, config, device)

def get_mnist_dataloaders(config: Config, device: torch.device)-> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load MNIST dataset and return training and test dataloaders.

    Args:
        config (Config): Configuration object containing dataset path and batch size.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test dataloaders.
    """
    # Mittelwert und Standardabweichung für MNIST

    pin_memory = False
    transform_mnist = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.1307,),std=(0.3081,))])

    # Load training dataset
    train_dataset = datasets.MNIST(root=config.dataset_path,train=True,download=True,transform=transform_mnist)

    # Load test dataset der 10.000 Bilder hat
    test_dataset = datasets.MNIST(root=config.dataset_path,train=False,download=True,transform=transform_mnist)

    return _create_dataloaders(train_dataset, test_dataset, config, device)

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

    transform_fashion = transforms.Compose([transforms.ToTensor(),transforms.Normalize((mean,), (std,))])

    train_dataset = datasets.FashionMNIST(root=config.dataset_path,train=True,download=True,transform=transform_fashion)

    test_dataset = datasets.FashionMNIST(root=config.dataset_path,train=False,download=True,transform=transform_fashion)

    return _create_dataloaders(train_dataset, test_dataset, config, device)

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

    transform_cifar10 = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])

    train_dataset = datasets.CIFAR10(root=config.dataset_path,train=True,download=True,transform=transform_cifar10)

    test_dataset = datasets.CIFAR10(root=config.dataset_path,train=False,download=True,transform=transform_cifar10)

    return _create_dataloaders(train_dataset, test_dataset, config, device)

def get_cifar100_dataloaders(config: Config, device: torch.device) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
       Load CIFAR-100 dataset and return training and test dataloaders.

       Args:
           config (Config): Configuration object containing dataset path and batch size.
           device (torch.device): Device to determine pin_memory setting.

       Returns:
           Tuple[DataLoader, DataLoader]: Training and test dataloaders.
       """
    # Mittelwert und Standardabweichung für CIFAR-100 (RGB-Kanäle)
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    if config.augment_data:
        transform_cifar100 = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, padding=4),
                                                 transforms.ToTensor(),transforms.Normalize(mean, std)])
    else:
        transform_cifar100 = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])

    train_dataset = datasets.CIFAR100(root= config.dataset_path,train=True,download=True,transform=transform_cifar100)

    test_dataset = datasets.CIFAR100(root= config.dataset_path,train=False,download=True,transform=transform_test)

    return _create_dataloaders(train_dataset, test_dataset, config, device)

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
