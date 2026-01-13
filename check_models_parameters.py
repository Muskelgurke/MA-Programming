import torchvision.models as models
import torch.nn as nn
import torch

def print_parameters(model: nn.Module):

    print("=" * 80)
    print(f"{model.__class__.__name__} Model Parameters")
    print("=" * 80)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

        print(f"{name:50s} | Shape: {str(param.shape):30s} | Params: {num_params:>10,}")
    print("\nLayer-Details:")
    print("=" * 80)
    for name, layer in model.named_modules():
        if name:  # Skip das Hauptmodell selbst
            print(f"{name}: {layer.__class__.__name__}")
    print(model)
    print("=" * 80)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    print("=" * 80)


# Call the function
if __name__ == "__main__":
    model = models.alexnet(pretrained=False)
    #model = models.vgg16(pretrained=False)
    #model = models.densenet121(pretrained=False)
    #model = models.efficientnet_v2_s(pretrained=False)
    #model = models.mobilenet_v2(pretrained=False)
    #model = models.resnet18(pretrained=False)
    #model = models.resnet34(pretrained=False)
    #model = models.resnet50(pretrained=False)
    #model = LeNet5(input_channels, num_classes, input_size)
    print_parameters(model)




class LeNet5(nn.Module):
    def __init__(self, num_input_channel, num_classes, input_size):
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

        feature_map_size = ((input_size-4)//2 -4)//2  # Berechnung der finalen Feature-Map-Größe nach den Convs und Pools
        flattend_size = 16 * feature_map_size * feature_map_size
        self.fc1 = nn.Linear(flattend_size, 120)
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
        y = self.fc2(y)
        y = self.activation4(y)
        y = self.fc3(y)
        return y