import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

'''MSUNet'''
class MSUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=6, wf=6, padding=True, norm='none', up_mode='upconv'):
        super(MSUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth

        n_channels_set = [64, 128, 256, 512, 512, 512, 512]
        self.down1 = MSUNetDownBlock(in_channels, n_channels_set[0], padding, norm)
        self.down2 = MSUNetDownBlock(n_channels_set[0], n_channels_set[1], padding, norm)
        self.down3 = MSUNetDownBlock(n_channels_set[1], n_channels_set[2], padding, norm)
        self.down4 = MSUNetDownBlock(n_channels_set[2], n_channels_set[3], padding, norm)
        self.down5 = MSUNetDownBlock(n_channels_set[3], n_channels_set[4], padding, norm)
        self.down6 = MSUNetDownBlock(n_channels_set[4], n_channels_set[5], padding, norm)
        self.down7 = MSUNetDownBlock(n_channels_set[5], n_channels_set[6], padding, norm)

        self.up7 = MSUNetUpBlock(n_channels_set[5], n_channels_set[5], up_mode, padding, norm)
        self.up6 = MSUNetUpBlock(n_channels_set[5]*2, n_channels_set[4], up_mode, padding, norm)
        self.up5 = MSUNetUpBlock(n_channels_set[4]*2, n_channels_set[3], up_mode, padding, norm)
        self.up4 = MSUNetUpBlock(n_channels_set[3]*2, n_channels_set[2], up_mode, padding, norm)
        self.up3 = MSUNetUpBlock(n_channels_set[2]*2, n_channels_set[1], up_mode, padding, norm)
        self.up2 = MSUNetUpBlock(n_channels_set[1]*2, n_channels_set[0], up_mode, padding, norm)
        self.up1 = MSUNetUpBlock(n_channels_set[0]*2, 32, up_mode, padding, norm)
        self.last = nn.Conv2d(32, out_channels, kernel_size=1)

        self.msdecoder4 = OutUpBlock(n_channels_set[2], 1, padding, norm, up_ratio=8)
        self.msdecoder3 = OutUpBlock(n_channels_set[1], 1, padding, norm, up_ratio=4)
        self.msdecoder2 = OutUpBlock(n_channels_set[0], 1, padding, norm, up_ratio=2)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u7 = self.up7(d7)
        u6 = self.up6(torch.cat((u7, center_crop(d6, u7.shape[2:])), 1))
        u5 = self.up5(torch.cat((u6, center_crop(d5, u6.shape[2:])), 1))
        u4 = self.up4(torch.cat((u5, center_crop(d4, u5.shape[2:])), 1))
        u3 = self.up3(torch.cat((u4, center_crop(d3, u4.shape[2:])), 1))
        u2 = self.up2(torch.cat((u3, center_crop(d2, u3.shape[2:])), 1))
        u1 = self.up1(torch.cat((u2, center_crop(d1, u2.shape[2:])), 1))
        out = self.last(u1)

        ms2 = self.msdecoder2(u2)
        ms3 = self.msdecoder3(u3)
        ms4 = self.msdecoder4(u4)

        out = out + ms2 + ms3 + ms4

        out = F.tanh(out)
        return out


##################################################################################
# Basic Functions
##################################################################################
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

##################################################################################
# Basic Blocks
##################################################################################
class MSUNetDownBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, norm):
        super(MSUNetDownBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=int(padding)))
        block.append(nn.LeakyReLU())
        if norm == 'BN':
            block.append(nn.BatchNorm2d(out_size))
        elif norm == 'IN':
            block.append(nn.InstanceNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class MSUNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, norm):
        super(MSUNetConvBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.LeakyReLU())
        if norm == 'BN':
            block.append(nn.BatchNorm2d(out_size))
        elif norm == 'IN':
            block.append(nn.InstanceNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class MSUNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm):
        super(MSUNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            block = []
            block.append(nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2))
            block.append(nn.LeakyReLU())
            if norm == 'BN':
                block.append(nn.BatchNorm2d(out_size))
            elif norm == 'IN':
                block.append(nn.InstanceNorm2d(out_size))
            self.up = nn.Sequential(*block)

        elif up_mode == 'upsample':
            block = []
            block.append(nn.Upsample(mode='bilinear', scale_factor=2))
            block.append(nn.Conv2d(in_size, out_size, kernel_size=1))
            block.append(nn.LeakyReLU())
            if norm == 'BN':
                block.append(nn.BatchNorm2d(out_size))
            elif norm == 'IN':
                block.append(nn.InstanceNorm2d(out_size))
            self.up = nn.Sequential(*block)

        self.conv_block = MSUNetConvBlock(out_size, out_size, padding, norm)

    def forward(self, x):
        up = self.up(x)
        out = self.conv_block(up)

        return out


def center_crop(layer, target_size):
    _, _, layer_height, layer_width = layer.size()
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]


class OutUpBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, norm, up_ratio):
        super(OutUpBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if norm == 'BN':
            block.append(nn.BatchNorm2d(out_size))
        elif norm == 'IN':
            block.append(nn.InstanceNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if norm == 'BN':
            block.append(nn.BatchNorm2d(out_size))
        elif norm == 'IN':
            block.append(nn.InstanceNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))

        block.append(nn.Upsample(mode='bilinear', scale_factor=up_ratio))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


if __name__ == '__main__':
    pass
