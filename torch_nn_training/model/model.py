import torch.nn as nn
import torch
from configuration.config_class import Config

def get_model(config: Config) -> nn.Module:
    match config.dataset_name.lower():
        case "mnist" | "fashionmnist":
            input_size = 28 * 28
            output_size = 10
            match config.model_type:
                case "fc256":
                    hidden_size = 256
                    return NN_fc(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
                case "fc1024":
                    hidden_size = 1024
                    return NN_fc(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
                case "fc4096":
                    hidden_size = 4096
                    return NN_fc(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
                case "cnn":
                    return CNN_for_MNIST()
                case _:
                    raise ValueError(f"Unknown model type: {config.model_type} for that Dataset")

        case "cifar10":
            input_size = 32 * 32 * 3
            output_size = 10
            match config.model_type:
                case "fc256":
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

        case "cifar100":
            input_size = 32 * 32 * 3
            output_size = 100
            match config.model_type:
                case "fc256":
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

        case "demo_linear_regression":
            # Add implementation for linear regression model
            input_size = 1  # Adjust based on your needs
            output_size = 1
            return linear_regression_model(input_size, output_size)
        case "small_mnist_for_manual_calculation":
            input_size = 4
            output_size = 2
            return _fc_manual_calculation(input_size, output_size)
        case _:
            raise ValueError(f"Unknown dataset: {config.dataset_name}")



class NN_fc(nn.Module):
    def __init__(self, input_size: int, hidden_size:int, output_size: int):
        super(NN_fc, self).__init__()

        # Original NN structure
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        # Original NN structure
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x

class linear_regression_model(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(linear_regression_model, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear.weight.data.fill_(1)
        self.linear.bias.data.fill_(2)

    def forward(self,x:torch.Tensor)-> torch.Tensor:
        x = self.linear(x)
        return x

    def get_weights_and_bias(self):
        """Return the weights (a) and bias (b) of the linear layer"""
        return self.linear.weight.data, self.linear.bias.data

class CNN_for_CIFAR(nn.Module):
    """Convolutional Neural Network for Cifar10/100 dataset
    aus dem Paper BAYDIN et al. 2022
    input 3x32x32 (3 Kanäle für RGB)
    """
    def __init__(self, output_size= 0):
        super(CNN_for_CIFAR, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second conv block
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64 x 8 x 8
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size) # Output: 10 Klassen
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNN_for_MNIST(nn.Module):
    """Convolutional Neural Network for MNIST/FashionMnist dataset
    aus dem Paper BAYDIN et al. 2022
    """
    def __init__(self):
        super(CNN_for_MNIST, self).__init__()

        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second conv block
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64 x 7 x 7
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10) # Output: 10 Klassen
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def _fc_manual_calculation(input_size: int, output_size: int) -> nn.Module:
    """Build fully connected model for manual calculation"""
    return fc_manual_calculation_model(input_size, output_size)

class fc_manual_calculation_model(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(fc_manual_calculation_model, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc1.weight.data = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                             [0.5, 0.6, 0.7, 0.8]], dtype=torch.float32)
        self.fc1.bias.data = torch.tensor([0.5, -0.5], dtype=torch.float32)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        return x

