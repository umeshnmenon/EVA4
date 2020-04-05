# -*- coding: utf-8 -*-
"""
Contains utility functions
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import *
from gradcam.utils import visualize_cam
from gradcam.gradcam import GradCAM
from torchvision.utils import make_grid, save_image
import torch
import torchvision.transforms.functional as F
from torchvision import datasets, transforms


def denormalize(tensor, mean, std):
  """
  Unnormalize and bring back the image
  """
  single_img = False
  if tensor.ndimension() == 3:
    single_img = True
    tensor = tensor[None,:,:,:]

  if not tensor.ndimension() == 4:
      raise TypeError('tensor should be 4D')

  mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
  std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
  ret = tensor.mul(std).add(mean)
  return ret[0] if single_img else ret


def unnormalize(img, mean, sd):
    """
    Unnormalize and bring back the image
    """
    img = img.numpy().astype(dtype=np.float32)
    print(img.shape[1])
    for i in range(img.shape[1]):
        img[i] = np.asarray(img[i], dtype=float)
        print(img[i].shape)
        img[i] = (img[i] * sd[i]) + mean[i]

    return np.transpose(img, (1, 2, 0))


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

