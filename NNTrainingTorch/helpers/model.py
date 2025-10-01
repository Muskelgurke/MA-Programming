import torch.nn as nn
import torch


from NNTrainingTorch.helpers.config_class import Config

def get_model(config: Config) -> nn.Module:
    match config.dataset_name.lower():
        case "mnist" | "fashionmnist":
            output_size = 10  # number of classes

            match config.model_type:
                case "fc":
                    return _build_fc_model(output_size)
                case "conv":
                    return _build_conv_model(output_size)
                case _:
                    raise ValueError(f"Unknown model type: {config.model_type} for that Dataset")

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

def _fc_manual_calculation(input_size: int, output_size: int) -> nn.Module:
    """Build fully connected model for manual calculation"""
    return fc_manual_calculation_model(input_size, output_size)


def _build_fc_model(output_size: int) -> nn.Module:
    """Build fully connected model"""
    input_size = 28 * 28  # picture size of MNIST only relevant for fully connected NN
    # number of neurons in hidden layer
    # Flügel nimmt -> hidden_size= [256,1024,4096]
    hidden_size = 256
    return NeuralNetwork_for_MNIST(input_size, hidden_size, output_size)

def _build_conv_model(output_size: int) -> nn.Module:
    """Build convolutional model"""
    input_channels = 1    # number of channels (1 for grayscale)
    return ConvNet_for_MNIST(input_channels, output_size)

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

class NeuralNetwork_for_MNIST(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(NeuralNetwork_for_MNIST, self).__init__()
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


class ConvNet_for_MNIST(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        """Standard Convolutional Network layers for the MNIST dataset.

                Args:
                    input_size (int): input size of the model.
                    output_size (int): The number of output classes.

        """
        super(ConvNet_for_MNIST,self).__init__()
        self.conv1 = torch.nn.Conv2d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(3136, 1024)
        self.fc2 = torch.nn.Linear(1024, output_size)

    def forward(self,x:torch.Tensor)-> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

