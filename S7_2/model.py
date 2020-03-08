# -*- coding: utf-8 -*-
"""
This file contains architecture of a DNN using CNN with Depthwise Separable Convolution and Dilated Convolution for
CIFAR10 data image classification
"""

# Commented out IPython magic to ensure Python compatibility.
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# %matplotlib inline
import matplotlib.pyplot as plt
import torchvision
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolution Block 1
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, dilation=2, bias=False),
                                   nn.ReLU(), nn.BatchNorm2d(32), nn.Dropout2d(0.03))
        # output_size = 32, RF = 3
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
                                   nn.ReLU(), nn.BatchNorm2d(32), nn.Dropout2d(0.03))
        # output_size = 32, RF = 5
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
                                   nn.ReLU(), nn.BatchNorm2d(32), nn.Dropout2d(0.03))
        # output_size = 32, RF = 7
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16, RF = 8
        
        # Convolution Block 2
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, groups=32, bias=False),
                                   nn.ReLU(), nn.BatchNorm2d(64), nn.Dropout2d(0.03))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                   nn.ReLU(), nn.BatchNorm2d(64), nn.Dropout2d(0.03))
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                   nn.ReLU(), nn.BatchNorm2d(64), nn.Dropout2d(0.03))
        self.pool2 = nn.MaxPool2d(2, 2) # 8

        # Convolution Block 3
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
                                   nn.ReLU(), nn.BatchNorm2d(128), nn.Dropout2d(0.03))
        self.conv8 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
                                   nn.ReLU(), nn.BatchNorm2d(128), nn.Dropout2d(0.03))
        self.conv9 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
                                   nn.ReLU(), nn.BatchNorm2d(128), nn.Dropout2d(0.03))
        self.pool3 = nn.MaxPool2d(2, 2) # 4

        # Convolution Block 4
        self.conv10 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1)
        self.gap = nn.AvgPool2d(3)

        # Fully Connected Block
        self.fc1 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool2(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.pool3(x)
        
        x = self.conv10(x)
        x = self.gap(x)
        x = x.view(-1, 32)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
