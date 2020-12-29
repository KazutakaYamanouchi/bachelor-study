from typing import Tuple, Union

import torch
import torch.nn as nn


def init_xavier_uniform(layer):
    if hasattr(layer, "weight"):
        torch.nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if hasattr(layer.bias, "data"):
            layer.bias.data.fill_(0)


class GBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        num_classes: int = 0
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
        self.conv.apply(init_xavier_uniform)

    def forward(self, x):
        x0 = self.upsample(x)
        x1 = self.conv1(x0)
        # TODO: Pixel-wise Normalization
        x2 = self.activation(x1)
        x3 = self.conv2(x2)
        x4 = self.activation(x3)
        return x4


class GFirst(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        num_classes: int = 0
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=4,
            stride=stride, padding=3, bias=False)

        self.activation = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False)
        self.conv.apply(init_xavier_uniform)

    def forward(self, x):
        x1 = self.conv1(x)
        # TODO: Pixel-wise Normalization
        x2 = self.activation(x1)
        x3 = self.conv2(x2)
        x4 = self.activation(x3)
        return x4


class Generator(nn.Module):
    def __init__(
        self, nz: int, nc: int, ngf: int = 64,
        num_classes: int = 0
    ):
        super().__init__()
        # (1, 1) -> (4, 4)
        self.block1 = GFirst(nz, ngf * 8)
        # (4, 4) -> (8, 8)
        self.block2 = GBlock(ngf * 8,  ngf * 8)
        # toRGB
        self.toRGB = nn.Sequential(
            nn.Conv2d(ngf * 8, nc, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, z):
        # z = z.view(-1, z.size(1), 1, 1)
        z1 = self.block1(z)
        z2 = self.block2(z1)
        return self.toRGB(z2)
