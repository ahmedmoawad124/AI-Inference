import torch
from torchvision.models import resnet18, mobilenet_v2, mobilenet_v3_small, regnet_y_8gf


class VanillaCNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(VanillaCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Input channels = 3 (RGB)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions by half
        self.fc1 = torch.nn.Linear(128 * 28 * 28, 256)  # Adjusted for 224x224 input size
        self.fc2 = torch.nn.Linear(256, num_classes)    # Final output layer
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(self.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = self.pool(self.relu(self.conv3(x)))  # Conv3 -> ReLU -> Pool
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.relu(self.fc1(x))  # Fully connected 1 -> ReLU
        logits = self.fc2(x)
        softmax_out = self.softmax(logits)

        return logits, softmax_out


class ResNet18Model(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Model, self).__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.model(x)
        softmax_out = self.softmax(logits)
        return logits, softmax_out


class MobileNetV2Model(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2Model, self).__init__()
        self.model = mobilenet_v2(pretrained=True)
        self.model.classifier[1] = torch.nn.Linear(self.model.last_channel, num_classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.model(x)
        softmax_out = self.softmax(logits)
        return logits, softmax_out
