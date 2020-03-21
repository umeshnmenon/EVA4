"""
This file keeps all the transformations tht are used in the pre-processing of the images

"""
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, Cutout, Rotate, GaussianBlur, VerticalFlip
from albumentations.pytorch import ToTensor, ToTensorV2
import numpy as np



class AlbuCompose:
    """
    Converts to PyTorch compatible Transform
    """
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self, img):
        img = np.array(img)
        img = self.transforms(image=img)["image"]
        return img


# Albumentations Transformations
transform_train_albu = Compose([
    RandomCrop(height=32, width=32), #, always_apply=True
    HorizontalFlip(p=0.2),
    VerticalFlip(p=0.1),
    GaussianBlur(p=0.0),
    Rotate(limit=15),
    #ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), always_apply=True),
    Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=[0.4914, 0.4822, 0.4465], p=0.25),
    ToTensorV2(always_apply=True)
])


transform_test_albu = Compose([
    #ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ToTensorV2(always_apply=True)
])

transform_test_albu = AlbuCompose(transform_test_albu)
transform_train_albu = AlbuCompose(transform_train_albu)
