from typing import Tuple, Union

import torch
import torch.nn as nn


def init_xavier_uniform(layer):
    if hasattr(layer, "weight"):
        torch.nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if hasattr(layer.bias, "data"):
            layer.bias.data.fill_(0)


class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    def __repr__(self):
        return self.__class__.__name__


class GBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False)

        self.activation = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False)

        self.pixelnorm = PixelNormLayer()

        self.conv1.apply(init_xavier_uniform)
        self.conv2.apply(init_xavier_uniform)

    def forward(self, x):
        x0 = self.upsample(x)
        x1 = self.conv1(x0)
        x2 = self.pixelnorm(x1)
        x3 = self.activation(x2)
        x4 = self.conv2(x3)
        x5 = self.pixelnorm(x4)
        x6 = self.activation(x5)
        return x6


class GFirst(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=4,
            stride=stride, padding=3, bias=False)

        self.activation = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False)

        self.pixelnorm = PixelNormLayer()

        self.conv1.apply(init_xavier_uniform)
        self.conv2.apply(init_xavier_uniform)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.activation(x1)
        x3 = self.conv2(x2)
        x4 = self.pixelnorm(x3)
        x5 = self.activation(x4)
        return x5


class Generator(nn.Module):
    def __init__(
        self, nz: int, num_progress: int,
        ngf: int = 64
    ):
        super().__init__()
        # (1, 1) -> (4, 4)
        self.block1 = GFirst(nz, ngf * 8)
        # (4, 4) -> (8, 8)
        self.block2 = GBlock(ngf * 8,  ngf * 8)
        # toRGB
        self.toRGB = nn.Conv2d(ngf * 8, 3, kernel_size=1, stride=1, padding=0)

        self.layer = (nn.Module)
        self.mod_list = nn.ModuleList()
        self.mod_list.append(self.block1)
        for i in range(num_progress):
            self.mod_list.append(self.block2)

    def forward(self, z):
        for i in range(len(self.mod_list)):
            z = self.mod_list[i](z)
        z1 = self.toRGB(z)
        return z1
