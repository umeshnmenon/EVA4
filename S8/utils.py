# -*- coding: utf-8 -*-
"""
Contains utility functions
"""
import matplotlib.pyplot as plt
import numpy as np
import torch


def imshow(img):
    """
    # functions to show an image
    :param img:
    :return:
    """
    img = img / 2 + 0.5     # denormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def get_device():
    """
    Get the GPU or CPU device
    :return: 
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device