"""
get data loaders
"""
from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision import transforms
import torch

def get_data_folder(dataset='imagenet'):
    """
    return the path to store the data
    """
    data_folder = os.path.join('/home/devuser/Downloads/data', dataset)

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class ImageFolderSample(datasets.ImageFolder):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """
    def __init__(self, root, transform=None, target_transform=None,
                 is_sample=False, k=4096):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        self.k = k
        self.is_sample = is_sample

        print('stage1 finished!')

        if self.is_sample:
            num_classes = len(self.classes)
            num_samples = len(self.samples)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                path, target = self.imgs[i]
                label[i] = target

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]

        print('dataset initialized!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            # sample_idx = np.hstack((neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


class CustomImageNet(datasets.ImageFolder):
    def __init__(self, root, transform=None, transform_list=None):

        super(CustomImageNet, self).__init__(root=root, transform=transform)
        self.transform_list = transform_list

    def __getitem__(self, index):

        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform_list is not None:
            img_transformed = []
            for transform in self.transform_list:
                img_transformed.append(transform(img.copy()))
            img = torch.stack(img_transformed)
        else:
            img = self.transform(img)
        return img, target

def get_test_loader(dataset='imagenet', batch_size=128, num_workers=8):
    """get the test data loader"""

    if dataset == 'imagenet':
        data_folder = get_data_folder(dataset)
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_folder = os.path.join(data_folder, 'val')
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    return test_loader


def get_imagenet_dataloader(dataset='imagenet', batch_size=128, num_workers=16, 
                            multiprocessing_distributed=False):
    """
    Data Loader for imagenet
    """
    train_transforms = []
    if dataset == 'imagenet':
        data_folder = get_data_folder(dataset)
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for i in range(2):
        transform_list = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ]
        train_transform = transforms.Compose(transform_list)
        train_transforms.append(train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'val')

    train_set = CustomImageNet(train_folder, transform_list=train_transforms)
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)

    if multiprocessing_distributed:
        train_sampler = DistributedSampler(train_set)
        test_sampler = DistributedSampler(test_set, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=num_workers,
                              pin_memory=True,
                              sampler=train_sampler)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             sampler=test_sampler)

    return train_loader, test_loader, train_sampler
