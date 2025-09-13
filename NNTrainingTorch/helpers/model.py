import torch.nn as nn

from NNTrainingTorch.helpers.Config import Config

def get_model(config: Config) -> nn.Module:
    if config.dataset.lower() in ["mnist", "fashionmnist"]:
        input_size = 28 * 28 # picture size
        hidden_size = 128 # number of neurons in hidden layer
        output_size = 10 # number of classes


        return NeuralNetwork_for_MNIST(input_size, hidden_size, output_size)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")



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



