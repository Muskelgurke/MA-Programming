import torch
import torchvision.models as models
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_input_channel, num_classes, input_size):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channel, 6, 5)
        self.activation1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.activation2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        feature_map_size = ((input_size - 4) // 2 - 4) // 2
        flattend_size = 16 * feature_map_size * feature_map_size
        self.fc1 = nn.Linear(flattend_size, 120)
        self.activation3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.activation4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        y = self.conv1(x)
        y = self.activation1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.activation2(y)
        y = self.pool2(y)
        y = torch.flatten(y, 1)
        y = self.fc1(y)
        y = self.activation3(y)
        y = self.fc2(y)
        y = self.activation4(y)
        y = self.fc3(y)
        return y

# Modell erstellen (z.B. AlexNet)
model = LeNet5(num_input_channel=1, num_classes=10, input_size=28)

model = model.cuda()

print(f"DEBUG: Model dtype is {next(model.parameters()).dtype}")
print(f"DEBUG: Allocated Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")




