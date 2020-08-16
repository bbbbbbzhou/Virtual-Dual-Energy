import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, norm_layer='IN'):
        super(Discriminator, self).__init__()

        nf = 64
        model = []
        model += [LeakyReLUConv2d(in_channels, nf, kernel_size=4, stride=2, padding=1)]
        model += [LeakyReLUConv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1, norm=norm_layer)]
        model += [LeakyReLUConv2d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1, norm=norm_layer)]
        model += [LeakyReLUConv2d(nf * 4, nf * 8, kernel_size=4, stride=1, norm=norm_layer)]
        model += [nn.Conv2d(nf * 8, 1, kernel_size=1, stride=1, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)

        return out


class Dis(nn.Module):
    def __init__(self, input_dim, n_layer=3, norm='None', sn=False):
        super(Dis, self).__init__()
        ch = 64
        self.model = self._make_net(input_dim, ch, n_layer, norm, sn)

    def _make_net(self, input_dim, ch, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=4, stride=2, padding=1, norm='None', sn=sn)]
        tch = ch
        for i in range(n_layer-1):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=4, stride=2, padding=1, norm=norm, sn=sn)]
            tch *= 2
        if sn:
            pass
        else:
            model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]
        return nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


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
class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            pass
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        elif norm == 'Batch':
            model += [nn.BatchNorm2d(n_out)]
        model += [nn.LeakyReLU(0.2, inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    pass
