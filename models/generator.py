from typing import Tuple, Union
import torch
import torch.nn as nn


def init_xavier_uniform(layer):
    if hasattr(layer, "weight"):
        torch.nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if hasattr(layer.bias, "data"):
            layer.bias.data.fill_(0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        print('initialize')


class GBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 4,
        stride: Union[int, Tuple[int, int]] = 2,
        padding: Union[int, Tuple[int, int]] = 1
    ):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=True))
        self.activation = nn.LeakyReLU(0.2)
        self.conv1.apply(init_xavier_uniform)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.activation(x1)
        return x2


class GFirst(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 4,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0
    ):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=True))
        self.activation = nn.LeakyReLU(0.2)
        self.conv1.apply(init_xavier_uniform)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.activation(x1)
        return x2


class Generator(nn.Module):
    def __init__(
        self, nz: int, num_progress: int,
        ngf: int = 64
    ):
        super().__init__()
        # toRGB
        self.toRGB = nn.Conv2d(
            ngf * 2 ** (3 - num_progress),
            3, kernel_size=1, stride=1, padding=0, bias=True)
        self.toRGB.apply(init_xavier_uniform)
        self.mod_list = nn.ModuleList()
        self.mod_list.append(GFirst(nz, ngf * 8))
        self.tanh = nn.Tanh()
        if num_progress != 0:
            num_progress = num_progress - 1
        for i in range(num_progress):
            self.mod_list.append(GBlock(
                ngf * 2 ** (3 - i), ngf * 2 ** (2 - i)))

    def progress(self, ngf, np):
        self.mod_list.append(GBlock(ngf * 2 ** (3 - np), ngf * 2 ** (2 - np)))

    def forward(self, z):
        x0 = z.view(-1, z.size(1), 1, 1)
        for i in range(len(self.mod_list)):
            x0 = self.mod_list[i](x0)
        x1 = self.toRGB(x0)
        x2 = self.tanh(x1)
        return x2
