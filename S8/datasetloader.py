# -*- coding: utf-8 -*-
"""
This file contains handy data loading functions tol load and transform on CIFAR10 data
"""

# Commented out IPython magic to ensure Python compatibility.
from __future__ import print_function
import torch
from torchvision import datasets, transforms
import torchvision


# function to load train and test data
def get_dataloader(is_train, cuda):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=16)
    # Check shuffle=False for Test
    dataset = torchvision.datasets.CIFAR10(root='./data', train=is_train, download=True,
                                           transform=transform_train if is_train else transform_test)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_args)
    return dataloader
