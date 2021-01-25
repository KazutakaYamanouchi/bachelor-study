# import torch.fft
import torch
from PIL import Image
import numpy as np


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None))
    if i != axis else slice(0, n, None)
        for i in range(X.dim())
            b_idx = tuple(slice(None, None, None))
    if i != axis else slice(n, None, None)
        for i in range(X.dim())
            front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def fftshift(X):
    real, imag = X.chunk(chunks=2, dim=-1)
    real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)
    for dim in range(2, len(real.size())):
        real = roll_n(real, axis=dim, n=int(np.ceil(real.size(dim) / 2)))
        imag = roll_n(imag, axis=dim, n=int(np.ceil(imag.size(dim) / 2)))
    real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    X = torch.cat((real, imag), dim=1)
    return torch.squeeze(X)


def ifftshift(X):
    real, imag = X.chunk(chunks=2, dim=-1)
    real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)
    for dim in range(len(real.size()) — 1, 1, -1):
        real = roll_n(real, axis=dim, n=int(np.floor(real.size(dim) / 2)))
        imag = roll_n(imag, axis=dim, n=int(np.floor(imag.size(dim) / 2)))
    real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    X = torch.cat((real, imag), dim=1)
    return torch.squeeze(X)


class FFT(nn.Module):
    def forward(self, image: torch.Tensor, k: int = 1) -> torch.Tensor:
        filter_rate = 1 - 1 / 2 ** k
        h, w = fft_shift_image.shape[:2] #画像の縦横
        cy, cx = int(h/2), int(w/2) # 画像の中心
        rh, rw = int(filter_rate * cy), int(filter_rate * cx) # フィルタサイズ
        fft_shift_img = fftshift(fft_img)
        fft_shift_img[cy-rh:cy+rh, cx-rw:cx+rw] = 0
