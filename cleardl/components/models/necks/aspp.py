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


class ConvBlock(nn.Sequential):
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


class Projection(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            SeparatableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates: list):
        super().__init__()
        r1, r2, r3, r4 = rates
        self.d1 = ConvBlock(r1, in_channels, out_channels)
        self.d2 = ConvBlock(r2, in_channels, out_channels)
        self.d3 = ConvBlock(r3, in_channels, out_channels)
        self.d4 = ConvBlock(r4, in_channels, out_channels)
        self.pool = PoolingBlock(1, in_channels, out_channels)
        self.proj = Projection(5*out_channels, out_channels)

        self._init_weights()

    def forward(self, x):
        _, _, H, W = x.size()
        d1 = self.d1(x)
        d2 = self.d2(x)
        d3 = self.d3(x)
        d4 = self.d4(x)
        p = F.interpolate(self.pool(x), size=(H, W), mode='bilinear', align_corners=True)
        out = torch.cat([d1, d2, d3, d4, p], dim=1)
        out = self.proj(out)
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
