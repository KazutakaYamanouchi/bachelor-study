import torch.fft
import torch.nn as nn


class FFT(nn.Module):
    def forward(self, images: torch.Tensor, k: int = 1) -> torch.Tensor:
        coefs = torch.fft.fft(images)  # 便宜上(B, C, H, W)とする
        # Low Pass Filter ( = High Frequency Cut)
        lpf_list = [1/16, 2/16, 4/16, 8/16, 12/16, 16/16]
        lpf_ratio = lpf_list[k]
        coefs[:, :, :coefs.size(2) * lpf_ratio * 0.5, :] = 0  # 上部分をゼロに
        coefs[:, :, coefs.size(2) * (1 - lpf_ratio) * 0.5:, :] = 0  # 下部分をゼロに
        coefs[:, :, :, :coefs.size(3) * lpf_ratio * 0.5] = 0  # 左部分をゼロに
        coefs[:, :, :, coefs.size(3) * (1 - lpf_ratio) * 0.5:] = 0  # 右部分をゼロに
        ifft_images = torch.fft.ifft(coefs)
        return ifft_images
