import torch
import torch.nn as nn
import numpy as np
from utils import arange
from networks.network_msunet import MSUNet
from networks.network_dis import Discriminator
import pdb


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])
    network = nn.DataParallel(network, device_ids=gpu_ids)

    return network


def get_generator(name, opts):
    if name == 'msunet':
        network = MSUNet(in_channels=1, norm='IN')
    else:
        raise NotImplementedError

    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters: {}'.format(num_param))
    return set_gpu(network, opts.gpu_ids)


def get_discriminator(name, opts):
    if name == 'patchGAN':
        network = Discriminator(in_channels=6, norm_layer='IN')
    else:
        raise NotImplementedError

    return set_gpu(network, opts.gpu_ids)


def get_gradoperator(opts):
    Kerx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Sobelx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    Sobelx.weight = nn.Parameter(torch.from_numpy(Kerx).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

    Kery = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Sobely = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    Sobely.weight = nn.Parameter(torch.from_numpy(Kery).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

    return set_gpu(Sobelx, opts.gpu_ids), set_gpu(Sobely, opts.gpu_ids)
