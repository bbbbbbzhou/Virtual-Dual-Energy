import os
import h5py
import random
import numpy as np
import pdb
import torch
import torchvision.utils as utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.data_patch_util import *
from PIL import Image
from scipy import ndimage, misc

import torchvision.transforms as transforms
import datasets.random_pair as random_pair


class DE_Train(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.IH_data_dir = os.path.join(self.root, 'Train', 'IH')
        self.IB_data_dir = os.path.join(self.root, 'Train', 'IB')
        self.IS_data_dir = os.path.join(self.root, 'Train', 'IS')
        self.data_files = sorted([os.path.join(self.IH_data_dir, f)
                                  for f in os.listdir(self.IH_data_dir) if f.endswith('.png')])

        self.AUG = opts.AUG

        transform_list = []
        transform_list.append(random_pair.randomrotate_pair(opts.angle))    # rotate the images
        self.transforms_rotate = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Scale(opts.osize, Image.BICUBIC))    # scale the images
        self.transforms_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(random_pair.randomcrop_pair(opts.fineSize))     # random crop image to a fineSize
        self.transforms_crop = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.ToTensor())
        self.transforms_toTensor = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Normalize([0.5], [0.5]))
        self.transforms_normalize = transforms.Compose(transform_list)

    def __getitem__(self, index):
        IH_filename = self.data_files[index]
        IB_filename = IH_filename.replace('IH', 'IB')
        IS_filename = IH_filename.replace('IH', 'IS')

        IH = Image.open(IH_filename).convert('L')
        IB = Image.open(IB_filename).convert('L')
        IS = Image.open(IS_filename).convert('L')

        if self.AUG:
            IH = self.transforms_scale(IH)
            IB = self.transforms_scale(IB)
            IS = self.transforms_scale(IS)
            [IH, IB, IS] = self.transforms_rotate([IH, IB, IS])
            [IH, IB, IS] = self.transforms_crop([IH, IB, IS])

        IH = self.transforms_toTensor(IH)
        IB = self.transforms_toTensor(IB)
        IS = self.transforms_toTensor(IS)

        IH = self.transforms_normalize(IH)
        IB = self.transforms_normalize(IB)
        IS = self.transforms_normalize(IS)

        return {'IH': IH,
                'IB': IB,
                'IS': IS}

    def __len__(self):
        return len(self.data_files)


class DE_Test(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.IH_data_dir = os.path.join(self.root, 'Test', 'IH')
        self.IB_data_dir = os.path.join(self.root, 'Test', 'IB')
        self.IS_data_dir = os.path.join(self.root, 'Test', 'IS')
        self.data_files = sorted([os.path.join(self.IH_data_dir, f)
                                  for f in os.listdir(self.IH_data_dir) if f.endswith('.png')])

        transform_list = []
        transform_list.append(transforms.ToTensor())
        self.transforms_toTensor = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Normalize([0.5], [0.5]))
        self.transforms_normalize = transforms.Compose(transform_list)

    def __getitem__(self, index):
        IH_filename = self.data_files[index]
        IB_filename = IH_filename.replace('IH', 'IB')
        IS_filename = IH_filename.replace('IH', 'IS')

        IH = Image.open(IH_filename).convert('L')
        IB = Image.open(IB_filename).convert('L')
        IS = Image.open(IS_filename).convert('L')

        IH = self.transforms_toTensor(IH)
        IB = self.transforms_toTensor(IB)
        IS = self.transforms_toTensor(IS)

        IH = self.transforms_normalize(IH)
        IB = self.transforms_normalize(IB)
        IS = self.transforms_normalize(IS)

        return {'IH': IH,
                'IB': IB,
                'IS': IS}

    def __len__(self):
        return len(self.data_files)


if __name__ == '__main__':
    pass
