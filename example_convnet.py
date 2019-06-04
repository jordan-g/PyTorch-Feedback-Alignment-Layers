import torch
import torch.nn as nn
from fa_layers import *

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class ConvNet(nn.Module):
    def __init__(self, input_channels):
        super(ConvNet, self).__init__()

        layers = []

        layers.append(Conv2dFA(input_channels, 64, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(Conv2dFA(64, 64, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(Flatten())
        layers.append(nn.Dropout2d(0.5))
        layers.append(LinearFA(1600, 384))
        layers.append(nn.ReLU(inplace=True))
        layers.append(LinearFA(384, 192))
        layers.append(nn.ReLU(inplace=True))
        layers.append(LinearFA(192, 10))
        layers.append(nn.Softmax())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

