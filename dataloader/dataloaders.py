import csv, torchvision, numpy as np, random, os
from PIL import Image
import torch
import copy
from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, Subset
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from collections import defaultdict
from .random_erase import RandomErasing
from .cutout import Cutout
import math
import random
from torchvision.transforms.autoaugment import AutoAugmentPolicy
# from torchvision.transforms import AugMix
from torchvision.datasets import CIFAR100
import sys

class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))

            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {
    }
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {
    i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {
    classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {
    }
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {
    classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {
    i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

class IdentityBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_instances, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            offset = k*self.batch_size%len(indices)
            batch_indices = indices[offset:offset+self.batch_size]

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                t = copy.deepcopy(self.dataset.classwise_indices[y])
                t.pop(t.index(idx))
                if len(t)>=(self.num_instances-1):
                    class_indices = np.random.choice(t, size=self.num_instances-1, replace=False)
                else:
                    class_indices = np.random.choice(t, size=self.num_instances-1, replace=True)
                pair_indices.extend(class_indices)

            yield batch_indices+pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // (self.batch_size)
        else:
            return self.num_iterations

class Custom_CIFAR100(CIFAR100):
    #------------------------
    #Custom CIFAR-100 dataset which returns returns 1 images, 1 target, image index
    #------------------------
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, [target, index]

class Custom_ImageFolder(ImageFolder):
    #------------------------
    #Custom ImageFolder dataset which returns 1 images, 1 target, image index
    #------------------------
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, [target, index]

class DatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        self.targets = dataset.targets
        self.classes = dataset.classes
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]

class TwoCropsTransform:
        """Take two random crops of one image as the query and key."""

        def __init__(self, base_transform):
            self.base_transform = base_transform

        def __call__(self, x):
            q = self.base_transform(x)
            k = self.base_transform(x)
            return [q, k]

class TwoCropsTransform2:
        """Take two random crops of one image as the query and key."""

        def __init__(self, transform1,transform2):
            self.base_transform = transform1
            self.aug_transform = transform2

        def __call__(self, x):
            q = self.base_transform(x)
            k = self.aug_transform(x)
            return [q, k]

from PIL import ImageFilter
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
def load_dataset(args):
    name = args.dataset
    root = args.data

    if name.startswith('CIFAR-100'):
        transforms_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        ]

        if args.data_aug == 'auto_aug':
            transforms_list.append(transforms.AutoAugment(AutoAugmentPolicy.CIFAR10))
        if args.data_aug == 'randaug':
            transforms_list.insert(0, transforms.RandAugment())
        if args.data_aug == 'augmix':
            transforms_list.append(transforms.AugMix())
        if args.data_aug == 'trivialaug':
            transforms_list.append(transforms.TrivialAugmentWide())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize([0.5071, 0.4867, 0.4408],[0.2675, 0.2565, 0.2761]))

        if args.data_aug == 'cutout': 
            transforms_list.append(Cutout(n_holes=1, length=8))
        if args.data_aug == 'random_erase':
            transforms_list.append(RandomErasing(mean=[0.5071, 0.4867, 0.4408]))
            
        transform_train = transforms.Compose(transforms_list)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        if args.method == 'DDGSD' :
            trainset = datasets.CIFAR100(root, train=True,  download=True, transform=TwoCropsTransform(transform_train))
        elif args.method == 'PSKD' or  args.method == 'dtskd':
            trainset = Custom_CIFAR100(root, train=True,  download=True, transform=transform_train)
        elif args.method == 'clsd' or  args.method == 'mixfrsd':
            trainset = datasets.CIFAR100(root, train=True,  download=True, transform=TwoCropsTransform(transform_train))
        else:
            trainset = datasets.CIFAR100(root, train=True,  download=True, transform=transform_train)
        
        valset   = datasets.CIFAR100(root, train=False, download=True, transform=transform_test)

    elif name.startswith('tinyimagenet'):
        transform_list = [
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]

        if args.data_aug == 'cutout': 
            transform_list.append(Cutout(n_holes=1, length=8))  
        transform_train = transforms.Compose(transform_list)

        transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

        train_val_dataset_dir = os.path.join(root, "train")
        test_dataset_dir = os.path.join(root, "val")
       
        trainset = TinyImageNet(root, train=True,transform=transform_train)
        valset = TinyImageNet(root, train=False,transform=transform_test)

    elif name in ['imagenet','CUB200', 'Dogs', 'MIT67', 'Air','Cars']:
        transforms_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ]
        if args.data_aug == 'auto_aug':
            transforms_list.append(transforms.AutoAugment(AutoAugmentPolicy.CIFAR10))
        if args.data_aug == 'randaug':
            transforms_list.insert(0, transforms.RandAugment())
          
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            
        transform_train = transforms.Compose(transforms_list)

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_val_dataset_dir = os.path.join(root, "train")
        test_dataset_dir = os.path.join(root, "val")

        if args.method == 'DDGSD' or args.method == 'stage':
            trainset = datasets.ImageFolder(root=train_val_dataset_dir, transform=TwoCropsTransform(transform_train))
        
        elif args.method == 'PSKD' or args.method == 'dtskd':
            trainset = Custom_ImageFolder(root=train_val_dataset_dir, transform=transform_train)
        else:
            trainset = datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train)
        
        valset = datasets.ImageFolder(root=test_dataset_dir, transform=transform_test)

    else:
        raise Exception('Unknown dataset: {}'.format(name))

    # Sampler
    if args.method.startswith('CS-KD') or args.method.startswith('psfrcsd'):
        get_train_sampler = lambda d: PairBatchSampler(d, args.batch_size)
        # get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), args.batch_size, False)

        trainset = DatasetWrapper(trainset)
        trainloader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=args.num_workers)
        # valloader = DataLoader(valset, batch_sampler=get_test_sampler(valset), num_workers=args.num_workers)

    elif args.method.startswith('BAKE'):
        get_train_sampler = lambda d: IdentityBatchSampler(d, args.batch_size, args.intra_imgs+1)
        trainset = DatasetWrapper(trainset)
        trainloader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=args.num_workers)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=(torch.cuda.is_available()))
    
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=(torch.cuda.is_available()))
    return trainloader, valloader
