import torch.utils
import torch.utils.data
from tqdm import tqdm

from PIL import Image

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, utils, transforms

from datasets import CityscapesDataset


def get_cityscape_data(
    mode,
    split,
    relabelled,
    root_dir='dataset/cityscapes',
    target_type='semantic',
    transforms=None,
    batch_size=1,
    eval=False,
    shuffle=True,
    pin_memory=True,
):
    data = CityscapesDataset(
        mode=mode,
        split=split,
        target_type=target_type,
        relabelled=relabelled,
        transform=transforms,
        root_dir=root_dir,
        eval=eval,
    )

    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=16,
    )

    return data_loader


def save_as_images(tensor_pred, folder, image_name, multiclass=True):
    tensor_pred = transforms.ToPILImage()(tensor_pred.byte())
    filename = f"{folder}/{image_name}.png"
    tensor_pred.save(filename)
