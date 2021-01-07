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
        padding: Union[int, Tuple[int, int]] = 1
    ):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False)

        self.activation = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False)

        self.conv1.apply(init_xavier_uniform)
        self.conv2.apply(init_xavier_uniform)

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
        padding: Union[int, Tuple[int, int]] = 1
    ):
        super().__init__()
        self.activation = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False)

        self.conv2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=4,
            stride=stride, padding=0, bias=False)

        self.linear = nn.Linear(in_channels, out_channels)

        self.conv1.apply(init_xavier_uniform)
        self.conv2.apply(init_xavier_uniform)
        self.linear.apply(init_xavier_uniform)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.activation(x1)
        x3 = self.conv2(x2)
        x4 = self.activation(x3)
        x5 = x4.view(x4.size()[0], -1)
        x6 = self.linear(x5)
        return x6


class Discriminator(nn.Module):
    def __init__(
        self, num_progress: int,
        ndf: int = 64
    ):
        super().__init__()
        # fromRGB
        self.fromRGB = nn.Sequential(
            nn.Conv2d(
                3, ndf * 8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2)
        )
        # (8, 8) -> (4, 4)
        self.block1 = DBlock(ndf * 8, ndf * 8)
        # (4, 4) -> (1, 1)
        self.block2 = DLast(ndf * 8, 1)

        self.layer = (nn.Module)
        self.mod_list = nn.ModuleList()
        for i in range(num_progress):
            self.mod_list.append(self.block1)
        self.mod_list.append(self.block2)

    def forward(self, z):
        z1 = self.fromRGB(z)
        for i in range(len(self.mod_list)):
            z1 = self.mod_list[i](z1)
        return z1
