import torch.utils.data
import torchaudio
import torchaudio.transforms as T
import kagglehub
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchaudio import datasets as datasets_audio
from helpers.config_class import Config
import scipy

# torchaudio.set_audio_backend("soundfile")

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

        case "cars":
            train_loader, test_loader = get_standCars_dataloaders(config, device)

        case "flower":
            train_loader, test_loader = get_flower102_dataloaders(config, device)

        case "food":
            train_loader, test_loader = get_food101_dataloaders(config, device)

        case "pet":
            train_loader, test_loader = get_oxfordPet_dataloaders(config,device)

        case "yesno":
            # TorchAudio
            raise ValueError(f"Unknown dataset-name: {config.dataset_name}")

        case "speechcommands":
            # TorchAudio
            raise ValueError(f"Unknown dataset-name: {config.dataset_name}")

        case "daliact":
            # LOW Prio -> wäre cool
            # wichtiger als cwru
            raise ValueError(f"Unknown dataset-name: {config.dataset_name}")

        case "cwru":
            # MaschineDataset
            # LOW Prio -> wäre cool
            raise ValueError(f"Unknown dataset-name: {config.dataset_name}")

        case _:
            raise ValueError(f"Unknown dataset-name: {config.dataset_name}")

    return train_loader, test_loader

def _create_dataloaders(train_dataset: torch.utils.data.Dataset,
                        test_dataset: torch.utils.data.Dataset,
                        config: Config,
                        device: torch.device,
                        cache_to_gpu: bool = True) -> tuple[
    torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Erstellt Dataloaders mit automatischem Fallback-Mechanismus.
    Versucht GPU-Caching, fällt bei OOM aber auf RAM-Streaming zurück.
    """

    # Imports hier, falls noch nicht oben vorhanden
    import torch.multiprocessing
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError:
        pass

    # 1. Wir laden ALLES in den RAM (mit großen Batches für Speed)
    # Das hast du schon getestet, das ist schnell.
    print(f"DATASET-> Pre-loading data into RAM (Workers: 8)...")

    def load_to_ram(dataset):
        loader = DataLoader(dataset, batch_size=256, num_workers=8, shuffle=False)
        x_list, y_list = [], []
        for x, y in loader:
            x_list.append(x)
            y_list.append(y)
        return torch.cat(x_list), torch.cat(y_list)

    try:
        print(f"DATASET-> Loading Training Data...")
        train_x, train_y = load_to_ram(train_dataset)

        print(f"DATASET-> Loading Test Data...")
        test_x, test_y = load_to_ram(test_dataset)

        # 2. Versuch: Ab auf die GPU damit!
        if cache_to_gpu:
            print("DATASET-> Trying to move data to GPU...", end="")
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            print(" Success! 🚀")

            # Wenn wir hier sind, ist alles auf der GPU.
            # Finaler Loader braucht keine Worker mehr.
            train_loader = DataLoader(TensorDataset(train_x, train_y),
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      num_workers=0)
            test_loader = DataLoader(TensorDataset(test_x, test_y),
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     num_workers=0)
            return train_loader, test_loader

    except RuntimeError as e:
        # 3. FALLBACK: Wenn der GPU Speicher voll ist (OOM)
        if "out of memory" in str(e).lower() or "allocate" in str(e).lower():
            print("\nWARNUNG: GPU voll! Fallback auf RAM-Streaming. (Keine Sorge, Training läuft weiter).")

            train_loader = DataLoader(TensorDataset(train_x, train_y),
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      num_workers=0,  # Bei TensorDataset im RAM sind 0 Worker oft schneller
                                      pin_memory=False)

            test_loader = DataLoader(TensorDataset(test_x, test_y),
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     num_workers=0,
                                     pin_memory=False)

            return train_loader, test_loader
        else:
            raise e  # Anderer Fehler? Absturz.

    # Fallback für den Fall cache_to_gpu=False explizit gesetzt
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=config.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=config.batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)

    return train_loader, test_loader


def _create_dataloaders_v0(train_dataset: torch.utils.data.Dataset,
                        test_dataset: torch.utils.data.Dataset,
                        config: Config,
                        device: torch.device,
                        cache_to_gpu: bool = True) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """ Private helper function to create dataloaders from datasets.

    def get_dataset(dataloader: DataLoader) -> TensorDataset:
        print("DATASET-> Starting Loading entire dataset into memory...")
        stack_x = list()
        stack_y = list()
        for idx, (x, y) in enumerate(dataloader):
            stack_x.append(x)
            stack_y.append(y)
            if idx% 20 == 0:
                print(f"DATASET-> Loaded {idx}/{len(dataloader)} samples into memory...")
        print("\nDATASET->  Concatenating all samples...")
        x = torch.squeeze(torch.stack(stack_x), 1).to(device)
        y = torch.squeeze(torch.stack(stack_y), 1).to(device)

        dataset = TensorDataset(x, y)
        return dataset

    """
    def get_dataset(dataloader: DataLoader) -> TensorDataset:
        print(f"DATASET-> Starting Loading entire dataset into memory (Batch size: {dataloader.batch_size})...")
        stack_x = list()
        stack_y = list()

        # 1. Daten einsammeln (CPU)
        for idx, (x, y) in enumerate(dataloader):
            stack_x.append(x)
            stack_y.append(y)
            if idx % 20 == 0:
                print(f"DATASET-> Loaded chunk {idx}/{len(dataloader)}...", end='\r')

        print("\nDATASET-> Concatenating and moving to GPU...")

        # 2. Zusammenfügen (cat statt stack!)
        # cat verbindet entlang der Dimension 0 (Batch-Dimension)
        # Wir machen das erst auf der CPU, dann .to(device) -> Sicherer gegen Abstürze
        try:
            full_x = torch.cat(stack_x, dim=0).to(device)
            full_y = torch.cat(stack_y, dim=0).to(device)
        except RuntimeError as e:
            print(f"\nFATAL ERROR: Not enough GPU memory to hold the entire dataset! Error: {e}")
            # Fallback oder Abbruch hier behandeln
            raise e

        dataset = TensorDataset(full_x, full_y)
        return dataset

    print("DATASET-> Pre-loading Training Data...")
    train_dataset_on_gpu = get_dataset(DataLoader(train_dataset, batch_size=256,num_workers=8))
    print("DATASET-> Pre-loading Test Data...")
    test_dataset_on_gpu = get_dataset(DataLoader(test_dataset, batch_size=256,num_workers=8))
    print("DATASET-> Finished loading entire dataset into memory.")

    train_drop_last = len(train_dataset_on_gpu) > config.batch_size

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset_on_gpu,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               #pin_memory=True if device.type == 'cuda' else False,
                                               pin_memory = False,
                                               drop_last=train_drop_last
                                               )

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset_on_gpu,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              #pin_memory=True if device.type == 'cuda' else False,
                                              pin_memory=False,
                                              drop_last=False
                                              )



    return train_loader, test_loader

def get_audio_transform(sample_rate: int=16000, n_mels: int=64, target_length: int= 100):
    """Create Tranform pipeline for audio to mel-spectrogram conversion"""

    transform = transforms.Compose([T.MelSpectrogram(sample_rate=sample_rate,n_fft=1024,hop_length=512, n_mels=n_mels),
                                   T.AmplitudeToDB(),
                                   transforms.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 2 else x),# Add channel dimension
                                   transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x), # Convert grayscale to RGB
                                   transforms.Resize((224, 224)),  # Resize to standard vision model input
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ])
    return transform

def get_speechCom_dataloaders(config: Config, device: torch.device)-> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
        Load SpeechCommands dataset from TorchAudio and return training and test dataloaders.

        Args:
            config (Config): Configuration object containing dataset path and batch size.

        Returns:
            Tuple[DataLoader, DataLoader]: Training and test dataloaders.

        """
    audio_transform = get_audio_transform(sample_rate=16000, n_mels=64)
    data = datasets_audio.SPEECHCOMMANDS(root=config.dataset_path, download=True)

    full_dataset = SpeechCommandsDataset(data, audio_transform)

    train_ratio = 0.8
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset=full_dataset,lengths= [train_size, test_size])

    return _create_dataloaders(train_dataset, test_dataset, config, device)

def get_yesNo_dataloaders(config: Config, device: torch.device)-> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
        Load YesNo dataset from TorchAudio and return training and test dataloaders.

        Args:
            config (Config): Configuration object containing dataset path and batch size.

        Returns:
            Tuple[DataLoader, DataLoader]: Training and test dataloaders.

        """
    audio_transform = get_audio_transform(sample_rate=8000, n_mels=64)

    full_dataset = datasets_audio.YESNO(root=config.dataset_path,download=True)

    full_dataset = YesNoDataset(full_dataset, audio_transform)

    train_ratio = 0.8
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset=full_dataset,lengths= [train_size, test_size])

    return _create_dataloaders(train_dataset, test_dataset, config, device)

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
    path = config.dataset_path + "/sanford-cars-dataset"
    if not os.path.exists(path):
        path = kagglehub.dataset_download("stanford-cars-dataset")

    transform_cars = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    train_dataset = datasets.StanfordCars(root=config.dataset_path,split='train',download=True,transform=transform_cars)

    test_dataset = datasets.StanfordCars(root=config.dataset_path,split='test',transform=transform_cars)

    return _create_dataloaders(train_dataset, test_dataset, config, device)

def get_mnist_dataloaders(config: Config, device: torch.device)-> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load MNIST dataset and return training and test dataloaders.

    Args:
        config (Config): Configuration object containing dataset path and batch size.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test dataloaders.
    """
    mean = 0.1307
    std = 0.3081
    match config.model_type:
        case "densenet":
            transform_mnist = transforms.Compose([
                transforms.Resize((56, 56)),  # 2x von 28x28
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,))
            ])
        case "alexnet":
            transform_mnist = transforms.Compose([
                transforms.Resize((84, 84)),  # 3x von 28x28
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,))
            ])
        case "vgg16":
            transform_mnist = transforms.Compose([
                transforms.Resize((84, 84)),  # 3x von 28x28
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,))
            ])
        case _:
            transform_mnist = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,))
            ])

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

    match config.model_type:
        case "densenet":
            transform_fashion = transforms.Compose([
                transforms.Resize((56, 56)),  # 2x von 28x28
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,))
            ])
        case "alexnet":
            transform_fashion = transforms.Compose([
                transforms.Resize((84, 84)),  # 3x von 28x28
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,))
            ])
        case "vgg16":
            transform_fashion = transforms.Compose([
                transforms.Resize((84, 84)),  # 3x von 28x28
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,))
            ])
        case _:
            transform_fashion = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,))
            ])

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

    match config.model_type:
        case "alexnet":
            transform_cifar10 = transforms.Compose([
                transforms.Resize((64, 64)),  # 2x von 32x32
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        case _:
            transform_cifar10 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])


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

    match config.model_type:
        case "alexnet":
            transform_cifar100 = transforms.Compose([
                transforms.Resize((64, 64)),  # 2x von 32x32
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        case _:
            transform_cifar100 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    train_dataset = datasets.CIFAR100(root= config.dataset_path,train=True,download=True,transform=transform_cifar100)

    test_dataset = datasets.CIFAR100(root= config.dataset_path,train=False,download=True,transform=transform_cifar100)

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

class YesNoDataset(torch.utils.data.Dataset):
    """YesNo Dataset returns: transformed_waveform, label_index"""

    def __init__(self, audio_dataset, transform):
        self.dataset = audio_dataset
        self.transform = transform
        # Erstelle Mapping für alle möglichen Label-Kombinationen
        self._create_label_mapping()

    def _create_label_mapping(self):
        """Erstelle Mapping von Label-Tupeln zu Klassen-Indizes"""
        unique_labels = set()
        for idx in range(len(self.dataset)):
            # Laden der Labels ohne Index-Fehler zu vermeiden
            try:
                _, _, label = self.dataset[idx]
                unique_labels.add(tuple(label))
            except IndexError:
                # Handle potential issue with dataset indexing if len(self.dataset) is not the actual size
                continue

        # Sortiere für konsistentes Mapping
        # Im Falle von YESNO sind die Labels Tupel aus 1/0
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        # Für YESNO ist das Label-Tupel 'yes' oder 'no'. Der Index ist 0 oder 1.

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # waveform: [C, L], sample_rate: int, label: List[int]
        waveform, sample_rate, label_list = self.dataset[idx]

        # 1. Transformiere die Wellenform in ein Mel-Spektrogramm (224x224 RGB)
        transformed = self.transform(waveform)

        # 2. Konvertiere Label-Liste/Tupel zu Klassen-Index
        label_idx = self.label_to_idx[tuple(label_list)]

        # Das zurückgegebene Label muss ein Tensor sein
        return transformed, torch.tensor(label_idx, dtype=torch.long)

class SpeechCommandsDataset(torch.utils.data.Dataset):
    """SpeechCommand dataset returns: transformed_waveform, label_index"""

    def __init__(self, audio_dataset, transform):
        self.dataset = audio_dataset
        self.transform = transform
        self._create_label_mapping()

    def _create_label_mapping(self):
        """Erstelle Mapping von String-Label zu Klassen-Indizes"""
        # Holen Sie sich alle eindeutigen String-Labels
        labels = sorted(list(set(item[2] for item in self.dataset._walker)))

        # Erstellen Sie das Mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}

        # Das Attribut `self.dataset._walker` ist eine Liste von Pfaden,
        # die zur Abfrage der Labels dient.
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # waveform: [C, L], sample_rate: int, label: str, speaker_id: str, utterance_number: int
        waveform, sample_rate, label_str, speaker_id, utterance_number = self.dataset[idx]

        # 1. Transformiere die Wellenform in ein Mel-Spektrogramm (224x224 RGB)
        transformed = self.transform(waveform)

        # 2. Konvertiere String-Label zu Klassen-Index
        label_idx = self.label_to_idx[label_str]

        return transformed, torch.tensor(label_idx, dtype=torch.long)