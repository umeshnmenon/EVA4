# -*- coding: utf-8 -*-
"""
This file contains handy data loading functions tol load and transform on CIFAR10 data
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

# function to load train and test data
def get_dataloader(is_train, cuda):
    transform = transforms.Compose([transforms.RandomRotation(7),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=16)

    dataset = torchvision.datasets.CIFAR10(root='./data', train=is_train, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_args)
    return dataloader