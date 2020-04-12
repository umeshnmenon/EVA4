"""
DNN based on custom architecture
"""
import torch.nn as nn
from models.resnet import *

class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class A11DNN(nn.Module):

    def __init__(self):
        """
        This function defines the architecture
        """

        super(A11DNN, self).__init__()

        # preparatory level
        self.x1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Layer 1
        self.x2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.R1 = ResNetBlock(in_planes=128, planes=128)

        # Layer 2
        self.x3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        # Layer 3
        self.x4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.R2 = ResNetBlock(in_planes=512, planes=512)

        self.x5 = nn.Sequential(
            nn.MaxPool2d(4, 4)
        )

        self.x6 = nn.Sequential(
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x1 = self.x1(x)
        x2 = self.x2(x1)
        x3 = self.R1(x2)
        x4 = self.x3(x2 + x3)
        x5 = self.x4(x4)
        x6 = self.R2(x5)
        x7 = self.x5(x5 + x6)
        x8 = x7.view(-1, 512)
        x9 = self.x6(x8)

        return F.log_softmax(x9)