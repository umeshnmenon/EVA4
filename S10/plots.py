"""
All the plotting functions used in the benchmarking analyses can be found here. The output of the plot must be compatible
with the auto reporting module.
"""

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
import torch
from utils import *
from gradcam.utils import visualize_cam
from gradcam.gradcam import GradCAM
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as F
from torchvision import datasets, transforms


def plot_cam(img, mean, sd, device):
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


def plot_lines(*args, **kwargs):
    """
    Plots accuracy as a line
    """
    lbls = kwargs.get('labels')
    title = kwargs.get('title')
    save = kwargs.get('save')
    plot_df = pd.DataFrame(columns=lbls)
    # get the max length
    max_len = 0
    for i, data in enumerate(args):
        arr_len = len(data)
        if arr_len > max_len:
            max_len = arr_len

    for i, data in enumerate(args):
        # check the array length and fill the gap with max_len with None
        # This will not happen normally though
        arr_len = len(data)
        if arr_len < max_len:
            data = data + [None] * (max_len - arr_len)
        data_arr = np.array(data)
        # data_arr = data_arr[~np.isnan(data_arr)]
        plot_df.loc[:, lbls[i]] = data_arr

    # sns.distplot(plot_df, hist=False)
    ax = plot_df.plot.line()
    ax.set_title(title)
    if save:
        plt_name = title + ".png"  # + str(time.time()) + ".png"
        plt.savefig(plt_name)
    else:
        plt.show()


def plot_misclassified_images(mc_images, classes, mean, sd, labels, pred_labels, save=False, file_name="mc_images.png"):
    """
    Plots the missclassified images
    """
    figure = plt.figure(figsize=(10, 10))

    num_of_images = len(mc_images)

    for index in range(1, num_of_images + 1):
        #ax = plt.subplot(5, 5, index+1)
        img = mc_images[index - 1].cpu()
        if img.dim() == 2:  # single image H x W
            img = img.unsqueeze(0)
        if img.dim() == 3:  # single image
            if img.size(0) == 1:  # if single-channel, convert to 3-channel
                img = torch.cat((img, img, img), 0)
            img = img.unsqueeze(0)

        if img.dim() == 4 and img.size(1) == 1:  # single-channel images
            img = torch.cat((img, img, img), 1)

        img = denormalize(img, mean, sd)  # unnormalize
        plt.subplot(5, 5, index)
        plt.axis('off')

        plt.imshow(((np.transpose(img.numpy()[index], (1, 2, 0)) * 255).astype('uint8'))) #, ax=ax
        #imshow(torchvision.utils.make_grid(img.cpu()))

        plt.title("Predicted: %s\nActual: %s" % (pred_labels[index], labels[index]))

    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(file_name)