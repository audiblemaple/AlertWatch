import torch.nn as nn
from torchvision import models

class Network(nn.Module):
    def __init__(self, num_classes=None):  # Eyes: 24 = 12 points * 2 (x, y); Face: 136 = 68 * 2 (x, y)
        super().__init__()
        self.model_name = 'resnet18'
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
