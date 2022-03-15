import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bins: list):
        super().__init__()
        self.pool = PoolingBlock(1, in_channels, out_channels)
        self.bins = bins
        for bin in self.bins:
            setattr(self, f'ac{bin}', AtrousConvBlock(bin, in_channels, out_channels))

    def forward(self, x):
        _, _, H, W = x.size()
        outs = [F.interpolate(self.pool(x), size=(H, W), mode='bilinear', align_corners=True)]
        for bin in self.bins:
            y = getattr(self, f'ac{bin}')(x)
            outs.append(y)
        out = torch.cat(outs, dim=1)
        return out


class DeepLabV3PHead(nn.Module):
    def __init__(self, in_channels: int, low_in_channels: int, mid_channels: int, low_out_channels: int,
                 bins: list, n_classes: int, input_size: list):
        super().__init__()
        self.low = nn.Sequential(
            nn.Conv2d(low_in_channels, low_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp = ASPP(in_channels, mid_channels, bins)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d((len(bins)+1)*mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.UpsamplingBilinear2d([v//4 for v in input_size])
        self.conv3x3 = nn.Sequential(
            SeparatableConv2d(mid_channels+low_out_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.drop = nn.Dropout2d(0.1)
        self.cls_top = nn.Conv2d(mid_channels, n_classes, kernel_size=1)
        self.up2 = nn.UpsamplingBilinear2d(input_size)

        self._init_weights()

    def forward(self, x: torch.Tensor, low: torch.Tensor):
        low = self.low(low)
        x = self.aspp(x)
        x = self.conv1x1(x)
        x = self.up1(x)
        x = torch.cat([x, low], dim=1)

        x = self.conv3x3(x)
        x = self.drop(x)
        x = self.cls_top(x)
        out = self.up2(x)

        return out

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    if 'cls_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


class DeepLabV3PAuxHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, n_classes: int, input_size: list):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.drop = nn.Dropout2d(0.1)
        self.cls_top = nn.Conv2d(mid_channels, n_classes, kernel_size=1)
        self.up = nn.UpsamplingBilinear2d(input_size)

        self._init_weights()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.drop(x)
        x = self.cls_top(x)
        out = self.up(x)
        return out

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    if 'cls_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
