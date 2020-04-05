# -*- coding: utf-8 -*-
"""
This file contains handy data loading functions tol load and transform on CIFAR10 data
"""

# Commented out IPython magic to ensure Python compatibility.
from __future__ import print_function
import torch
import torchvision


def get_dataloader_CIFAR10(is_train, cuda=True, transform=None):
    """
    Gets the CIFAR10 dataset
    :return:
    """
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True,
                                                                                                           batch_size=16)

    # Check shuffle=False for Test
    dataset = torchvision.datasets.CIFAR10(root='./data', train=is_train, download=True,
                                           transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_args)

    return dataloader