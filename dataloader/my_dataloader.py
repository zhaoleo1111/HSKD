from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import Compose, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import os.path
import numpy as np
import sys
from torch.utils.data.distributed import DistributedSampler

import pickle
import torch
import torch.utils.data as data
from itertools import permutations
from .cutout import Cutout

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError(
                "Only transforms or transform/target_transform can "
                "be passed as argument"
            )

        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return ["{}{}".format(head, lines[0])] + [
            "{}{}".format(" " * len(head), line) for line in lines[1:]
        ]

    def extra_repr(self):
        return ""


class CIFAR10(VisionDataset):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self, root, train=True, transform=None, download=False, transform_list=None
    ):

        super(CIFAR10, self).__init__(root)
        self.transform = transform
        self.transform_list = transform_list
        self.train = train  # training set or test set

        if download:
            raise ValueError("cannot download.")
            exit()

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []


        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        # if not check_integrity(path, self.meta['md5']):
        #    raise RuntimeError('Dataset metadata file not found or corrupted.' +
        #                       ' You can use download=True to download it')
        with open(path, "rb") as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        if self.transform_list is not None:
            img_transformed = []
            for transform in self.transform_list:
                img_transformed.append(transform(Image.fromarray(img.copy())))
            img = torch.stack(img_transformed)
        else:
            img = self.transform(Image.fromarray(img))
        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None,transform_list=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")
        self.transform_list = transform_list
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
        # if self.transform is not None:
        #     sample = self.transform(sample)
            ###############
        if self.transform_list is not None:
            img_transformed = []
            for transform in self.transform_list:
                img_transformed.append(transform(sample.copy()))
            sample = torch.stack(img_transformed)
        else:
            sample = self.transform(sample)
        
        return sample, tgt

class CustomImageNet(ImageFolder):
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

def get_dataloader(args, ddp=False):
    train_transforms = []
    normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )
    if args.dataset == "CIFAR-100":
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )
    elif args.dataset == "CIFAR-10":
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        )
    elif args.dataset == "tinyimagenet":
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    elif args.dataset == "imagenet" or args.dataset == "CUB200" or  args.dataset == "Dogs" or  args.dataset == "MIT67" or  args.dataset == "Flower" or  args.dataset == "Air" or  args.dataset == "Cars":
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if args.dataset == "CIFAR-100" or args.dataset == "CIFAR-10":
        for i in range(2):
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            train_transforms.append(train_transform)
           
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    elif args.dataset == "imagenet" or args.dataset == "CUB200" or  args.dataset == "Dogs" or  args.dataset == "MIT67" or  args.dataset == "Flower" or  args.dataset == "Air" or args.dataset == "Cars":
        for i in range(2):
            transform_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ]
            train_transform = transforms.Compose(transform_list)
            train_transforms.append(train_transform)

        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

    elif args.dataset == "tinyimagenet":
        for i in range(2):
            # train_transform = transforms.Compose(
            #     [
            #         transforms.RandomResizedCrop(32),
            #         transforms.RandomHorizontalFlip(),
            #         transforms.ToTensor(),
            #         normalize,
            #     ]
            # )
            # train_transforms.append(train_transform)
            transform_list = [
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]

            if args.data_aug == 'cutout': 
                transform_list.append(Cutout(n_holes=1, length=8))  
            train_transform = transforms.Compose(transform_list)
            train_transforms.append(train_transform)

        test_transform = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), normalize]
        )
    
    if args.dataset == "CIFAR-100":
        trainset = CIFAR100(
            root=args.data, train=True, transform_list=train_transforms, download=False
        )
        valset = CIFAR100(
            root=args.data, train=False, transform=test_transform, download=False
        )
    elif args.dataset == "CIFAR-10":
        trainset = CIFAR10(
            root=args.data, train=True, transform_list=train_transforms, download=False
        )
        valset = CIFAR10(
            root=args.data, train=False, transform=test_transform, download=False
        )
    elif args.dataset == "tinyimagenet":
        trainset = TinyImageNet(
            root=args.data,train=True, transform_list=train_transforms
        )
        valset = TinyImageNet(
            root=args.data,train=False,transform=test_transform
        )
    elif args.dataset == "imagenet" or args.dataset == "CUB200" or  args.dataset == "Dogs" or  args.dataset == "MIT67" or  args.dataset == "Flower" or  args.dataset == "Air" or args.dataset == "Cars":
        trainset = CustomImageNet(root=os.path.join(args.data, "train"), transform_list=train_transforms)
        valset   = CustomImageNet(root=os.path.join(args.data, "val"), transform=test_transform)
    
    ###image net###
    # if multiprocessing_distributed:
    #     train_sampler = DistributedSampler(trainset,shuffle=True,)
    #     test_sampler = DistributedSampler(valset, shuffle=False)
    # else:
    #     train_sampler = None
    #     test_sampler = None

    # train_loader = torch.utils.data.DataLoader(trainset,
    #                           batch_size=args.batch_size,
    #                           shuffle=(train_sampler is None),
    #                           num_workers=args.num_workers,
    #                           pin_memory=True,
    #                           sampler=train_sampler)

    # val_loader = torch.utils.data.DataLoader(valset,
    #                          batch_size=args.batch_size,
    #                          shuffle=False,
    #                          num_workers=args.num_workers,
    #                          pin_memory=True,
    #                          sampler=test_sampler)

    if not ddp:
        train_loader = DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            valset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        # DistributedSampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainset,
            shuffle=True,
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            valset,
            shuffle=False,
        )
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=train_sampler,
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=val_sampler,
        )

    return train_loader, val_loader