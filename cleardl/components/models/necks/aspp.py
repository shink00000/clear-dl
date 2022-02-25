import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.separatable_conv import SeparatableConv2d


class PoolingBlock(nn.Sequential):
    def __init__(self, pool_size: int, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(output_size=pool_size),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class AtrousConvBlock(nn.Sequential):
    def __init__(self, rate: int, in_channels: int, out_channels: int):
        if rate == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            conv = SeparatableConv2d(in_channels, out_channels, kernel_size=3, padding=rate,
                                     dilation=rate, bias=False)
        super().__init__(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class OutBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, outs: tuple):
        out = torch.cat(outs, dim=1)
        return super().forward(out)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bins: list):
        super().__init__()
        self.pool = PoolingBlock(1, in_channels, out_channels)
        self.bins = bins
        for bin in self.bins:
            setattr(self, f'ac{bin}', AtrousConvBlock(bin, in_channels, out_channels))
        self.out_conv = OutBlock((len(self.bins)+1)*out_channels, out_channels)

        self._init_weights()

    def forward(self, x):
        _, _, H, W = x.size()
        outs = [F.interpolate(self.pool(x), size=(H, W), mode='bilinear', align_corners=True)]
        for bin in self.bins:
            y = getattr(self, f'ac{bin}')(x)
            outs.append(y)
        out = self.out_conv(outs)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
