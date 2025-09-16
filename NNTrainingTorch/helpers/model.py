import torch.nn as nn
import torch


from NNTrainingTorch.helpers.config_class import Config

def get_model(config: Config) -> nn.Module:
    if config.dataset_name.lower() in ["mnist", "fashionmnist"]:
        output_size = 10  # number of classes

        model_builders = {
            "fc":   lambda: _build_fc_model(output_size),
            "conv": lambda: _build_conv_model(output_size)
        }

        if config.model_type in model_builders:
            return model_builders[config.model_type]()
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")

def _build_fc_model(output_size: int) -> nn.Module:
    """Build fully connected model"""
    input_size = 28 * 28  # picture size only relevant for fully connected NN
    hidden_size = 128     # number of neurons in hidden layer
    return NeuralNetwork_for_MNIST(input_size, hidden_size, output_size)

def _build_conv_model(output_size: int) -> nn.Module:
    """Build convolutional model"""
    input_channels = 1    # number of channels (1 for grayscale)
    return ConvNet_for_MNIST(input_channels, output_size)


class NeuralNetwork_for_MNIST(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(NeuralNetwork_for_MNIST, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

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

