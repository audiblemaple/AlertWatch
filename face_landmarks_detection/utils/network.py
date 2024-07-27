"""
Module: Network

This module provides a class `Network` that defines a neural network model based on ResNet-18 architecture, customized for landmark detection tasks.

Classes:
    Network

Methods:
    __init__(self, num_classes=None)
        Initializes the Network class with the specified number of output classes.

    forward(self, x)
        Defines the forward pass of the network.
"""

import torch.nn as nn
from torchvision import models

class Network(nn.Module):
    """
    A class to define a neural network model based on ResNet-18 architecture,
    customized for landmark detection tasks.
    """
    def __init__(self, num_classes=None):
        """
        Initializes the Network class with the specified number of output classes.

        Args:
            num_classes (int, optional): The number of output classes.
                For eyes, it should be 24 (12 points * 2 for x, y).
                For face, it should be 136 (68 points * 2 for x, y).
        """
        super().__init__()
        self.model_name = 'resnet18'
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        x = self.model(x)
        return x
