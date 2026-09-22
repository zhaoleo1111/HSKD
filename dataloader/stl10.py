from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch
import math
from torch.utils.data import Dataset
import codecs
"""
mean = {
    'stl10': (0.485, 0.456, 0.406),
}

std = {
    'stl10': (0.229, 0.224, 0.225),
}
"""


def get_data_folder():
    """
    return the path to store the data
    """
    data_folder = '/home/devuser/Downloads/data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

def get_stl10_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """
    stl10
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomCrop(32, padding=4),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_set = datasets.STL10(root=data_folder,
                                      download=True,
                                      split='train',
                                      transform=train_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
  

    test_set = datasets.STL10(root=data_folder,
                                 download=True,
                                 split='test',
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))


    return train_loader, test_loader

