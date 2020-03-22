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


def plot_cam(img, mean, sd):
    """
    Plots the cam
    """
    # convert to PIL
    pil_img = F.to_pil_image(img)

    # normed_torch_img = transform_test_albu(pil_img).to(device)

    # call the transformation. To keep it simple, we are calling PyTorch way of transform
    torch_img = transforms.Compose([
        transforms.ToTensor()
    ])(pil_img).to(device)
    normed_torch_img = transforms.Normalize(mean=mean, std=sd)(torch_img)[None]

    # Call the GridCAM
    config = dict(model_type='resnet', arch=net, layer_name='layer4')
    gradcam = GradCAM.from_config(**config)

    images = []

    mask, _ = gradcam(normed_torch_img)
    heatmap, result = visualize_cam(mask, torch_img)

    images.extend([torch_img.cpu(), heatmap, result])

    grid_image = make_grid(images, nrow=5)

    return transforms.ToPILImage()(grid_image)