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


class DBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 4,
        stride: Union[int, Tuple[int, int]] = 2,
        padding: Union[int, Tuple[int, int]] = 1
    ):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=True))
        self.activation = nn.LeakyReLU(0.2)
        self.conv1.apply(init_xavier_uniform)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.activation(x1)
        return x2


class DLast(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 4,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0
    ):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=True))
        self.activation = nn.LeakyReLU(0.2)
        self.conv1.apply(init_xavier_uniform)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.activation(x1)
        return x2


class Discriminator(nn.Module):
    def __init__(
        self, num_progress: int,
        ndf: int = 64
    ):
        super().__init__()
        # fromRGB
        self.fromRGB = nn.Conv2d(
                3, ndf * 2 ** (3 - num_progress),
                kernel_size=1, stride=1, padding=0, bias=True)
        self.fromRGB.apply(init_xavier_uniform)
        self.mod_list = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()
        if num_progress != 0:
            num_progress = num_progress - 1
        for i in range(num_progress, 0, -1):
            self.mod_list.append(
                DBlock(ndf * 2 ** (3 - i),  ndf * 2 ** (4 - i)))
        self.mod_list.append(DLast(ndf * 8, 1))

    def progress(self, ndf, np):
        self.mod_list.append(DBlock(ndf * 2 ** (3 - np),  ndf * 2 ** (4 - np)))

    def forward(self, z):
        z1 = self.fromRGB(z)
        for i in range(len(self.mod_list)):
            z1 = self.mod_list[i](z1)
        z2 = self.sigmoid(z1)
        return z2
