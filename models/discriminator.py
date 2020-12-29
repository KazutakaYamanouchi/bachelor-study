from typing import Tuple, Union

import torch
import torch.nn as nn


def init_xavier_uniform(layer):
    if hasattr(layer, "weight"):
        torch.nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if hasattr(layer.bias, "data"):
            layer.bias.data.fill_(0)


class DBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        num_classes: int = 0
    ):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False)

        self.activation = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False)
        self.conv.apply(init_xavier_uniform)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.activation(x0)
        x2 = self.conv2(x1)
        x3 = self.activation(x2)
        x4 = self.downsample(x3)
        return x4


class DLast(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        num_classes: int = 0
    ):
        super().__init__()
        # self.conv0 = nn.Conv2d(
        #    in_channels, in_channels, kernel_size=kernel_size,
        #    stride=stride, padding=padding, bias=False)

        self.activation = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=4,
            stride=stride, padding=0, bias=False)

        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=stride, padding=0, bias=False)

        self.conv.apply(init_xavier_uniform)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.activation(x1)
        x3 = self.conv2(x2)
        x4 = self.activation(x3)
        return x4


class Discriminator(nn.Module):
    def __init__(
        self, nz: int, nc: int, ndf: int = 64,
        num_classes: int = 0
    ):
        super().__init__()
        # fromRGB
        self.fromRGB = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2)
        )
        # (8, 8) -> (4, 4)
        self.block1 = DBlock(ndf * 8, ndf * 8)
        # (4, 4) -> (1, 1)
        self.block2 = DLast(nz, 1)

    def forward(self, z):
        z1 = self.fromRGB(z)
        z2 = self.block1(z1)
        z3 = self.block2(z2)
        return z3
