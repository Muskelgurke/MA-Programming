import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F

from torchvision.models.resnet import BasicBlock, Bottleneck
from helpers.config_class import Config

def get_model(config: Config, sample_batch: tuple) -> nn.Module:
    """get model based on config and automatically set input and output sizes from sample batch
    Args:
        config: Config file
        sample_batch: tuple of (inputs, targets) from dataloader

    Returns:
        nn.Module: model
    """
    num_classes = 0
    inputs, targets = sample_batch

    input_channels = inputs.shape[1]
    input_size = inputs.shape[2]
    # for Torchaudio datasets (yes_no)
    if isinstance(targets, list):
        targets = torch.tensor(targets)

    # Bestimme Anzahl der Klassen aus dem gesamten Dataset, nicht nur aus dem Batch
    match config.dataset_name.lower():

        case "mnist": num_classes = 10
        case "fashionmnist": num_classes = 10
        case "cifar10": num_classes = 10
        case "cifar100":num_classes = 100
        case "flower": num_classes = 102
        case "food": num_classes = 101  # Food101 hat 101 Klassen
        case "pet": num_classes = 37  # OxfordPet hat 37 Klassen
        case _: num_classes = int(targets.max().item()) + 1

    specs = {
        "input_channels": input_channels,
        "input_size": input_size,
        "num_classes": num_classes
    }

    match config.model_type.lower():
        case "testmodel": return SimpleTestModel(input_channels, num_classes, input_size)
        case "alexnet": return get_adapted_model(models.alexnet ,specs)
        case "vgg16":   return get_adapted_model(models.vgg16 ,specs)
        case "resnet18":return get_adapted_model(models.resnet18 ,{**specs, "disable_inplace": True})
        case "resnet34":return get_adapted_model(models.resnet34 ,{**specs, "disable_inplace": True})
        case "resnet50":return get_adapted_model(models.resnet50 ,{**specs, "disable_inplace": True})
        case "mobilenet":return get_adapted_model(models.mobilenet_v2 ,specs)
        case "densenet":return get_adapted_model(models.densenet121 ,specs)
        case "efficientnet":return get_adapted_model(models.efficientnet_v2_s ,specs)
        case "lenet": return LeNet5(input_channels, num_classes)
        case _:
            raise ValueError(f"Unknown model type: {config.model_type}")

def get_adapted_model(model_fn= type[nn.Module], specs= dict):
    disable_inplace = specs.get("disable_inplace", False)
    model = model_fn(weights=None)
    if disable_inplace:
        # Wir iterieren durch ALLE Layer des Modells (inkl. BasicBlocks)
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
    if disable_inplace:
        # Wir überschreiben die Methoden in der Klasse selbst.
        # Das wirkt sich auf alle Instanzen aus, die danach (oder davor) erstellt wurden.
        BasicBlock.forward = patched_basic_block_forward
        Bottleneck.forward = patched_bottleneck_forward

    num_classes = specs.get("num_classes")
    input_channels = specs.get("input_channels")

    if num_classes is None:
        raise ValueError("specs must contain 'num_classes'")
    if input_channels is None:
        raise ValueError("specs must contain 'input_channels'")

    input_channels_are_not_3 = input_channels !=3 # for MNIST and fashionMNIST

    if input_channels_are_not_3:
        if hasattr(model, "classifier"):
            first_layer = model.features[0]
            # handle Conv2dNormActivation wrapper (efficientNet)
            if hasattr(first_layer, '0'):
                first_conv = first_layer[0]
            else:
                first_conv = first_layer

            new_conv = nn.Conv2d(in_channels=input_channels,
                                          out_channels= first_conv.out_channels,
                                          kernel_size= first_conv.kernel_size,
                                          stride= first_conv.stride,
                                          padding= first_conv.padding,
                                          bias=first_conv.bias is not None)

            if hasattr(first_layer, '0'):
                first_layer[0] = new_conv
            else:
                model.features[0] = new_conv

        elif hasattr(model, "conv1"):
            # For ResNet Models
            model.conv1 = nn.Conv2d(in_channels= input_channels,
                                    out_channels= model.conv1.out_channels,
                                    kernel_size= model.conv1.kernel_size,
                                    stride= model.conv1.stride,
                                    padding= model.conv1.padding,
                                    bias= False
            )

    # Output-Layer anpassen
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            # AlexNet, VGG
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            # MobileNet, DenseNet, EfficientNet
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        # ResNet-Modelle
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model

# --- Hilfsfunktion zum Patchen von ResNet BasicBlock ---
def patched_basic_block_forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    # WICHTIG für JVP: 'out = out + identity' statt 'out += identity'
    # sonst kommt error!!
    out = out + identity
    out = self.relu(out)

    return out

def patched_bottleneck_forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    # FIX: Out-of-place Addition für JVP/FGD
    out = out + identity
    out = self.relu(out)

    return out


class LeNet5(nn.Module):
    def __init__(self, num_input_channel=1, num_classes=10):
        super(LeNet5, self).__init__()
        activation = nn.modules.activation.ReLU
        # assume the input tensor of shape (None,num_input_channel,32,32);
        # None means we don't know mini-batch size yet, 1 is for one channel (grayscale)
        # num_input_channel is the number of input channels; 1 for grayscale images, and 3 for RBG color images
        self.conv1 = nn.Conv2d(num_input_channel, 6, 5)
        self.activation1 = activation()
        # after conv1 layer (5x5 convolution kernels, 6 output channels), the feature map is of shape (None,6,28,28)
        self.pool1 = nn.MaxPool2d(2)
        # after max pooling (kernel_size=2, so is stride), the feature map is of shape (None,6,14,14)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.activation2 = activation()
        # after conv1 layer (5x5 convolution kernels, 16 output channels), the feature map is of shape (None,16,10,10)
        self.pool2 = nn.MaxPool2d(2)
        # after max pooling (kernel_size=2, so is stride), the feature map is of shape (None,16,5,5)
        self.fc1 = nn.Linear(400, 120)
        self.activation3 = activation()
        # note that 16*5*5 = 400
        # the feature map of shape (None,16,5,5) is flattend and of shape (None,400),
        # followed by a fully-connected layer without 120 output features
        # after that, the feature map is now of shape (None,120)
        self.fc2 = nn.Linear(120, 84)
        self.activation4 = activation()
        # after fc2 layer, the feature map is of shape (None,84)
        self.fc3 = nn.Linear(84, num_classes)
        # after fc3 layer, the feature map is of shape (None,10)
        # we don't need softmax here (just raw network output),
        # since it will be taken care of in the routine torch.nn.CrossEntropyLoss
        # for more information, see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    def forward(self, x):
        # feed input x into LeNet-5 network (chain the layers) and get output y
        y = self.conv1(x)
        y = self.activation1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.activation2(y)
        y = self.pool2(y)
        y = torch.flatten(y, 1)  # flatten all dimensions except for the very first (batch dimension)
        y = self.fc1(y)
        y = self.activation3(y)
        if self.dropout1 is not None:
            y = self.dropout1(y)
        y = self.fc2(y)
        y = self.activation4(y)
        if self.dropout2 is not None:
            y = self.dropout2(y)
        y = self.fc3(y)
        return y

class SimpleTestModel(nn.Module):
    """
    Ein dynamisches, minimalistisches CNN für schnelles Debugging.
    Passt sich automatisch an Input-Größe und Channels an.
    """

    def __init__(self, input_channels, num_classes, input_size):
        super(SimpleTestModel, self).__init__()

        # Conv Layer: Behält HxW bei (durch padding=1 bei kernel=3)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=4, kernel_size=3, padding=1)

        # Pooling halbiert die Größe (H/2, W/2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Berechnung der Flatten-Größe für den Linear Layer
        # Nach MaxPool ist das Bild halb so groß
        final_dim = input_size // 2
        linear_input_size = 4 * final_dim * final_dim

        self.fc1 = nn.Linear(linear_input_size, num_classes)

    def forward(self, x):
        # x: [Batch, In_Channels, H, W]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)  # -> [Batch, 4, H/2, W/2]

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        return x
