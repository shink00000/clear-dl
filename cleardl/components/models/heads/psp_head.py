import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PoolingBlock(nn.Sequential):
    def __init__(self, pool_size: int, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(output_size=pool_size),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class PyramidPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bins: list):
        super().__init__()
        self.bins = bins
        for bin in self.bins:
            setattr(self, f'pc{bin}', PoolingBlock(bin, in_channels, out_channels))

    def forward(self, x):
        _, _, H, W = x.size()
        outs = [x]
        for bin in self.bins:
            y = getattr(self, f'pc{bin}')(x)
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
            outs.append(y)
        out = torch.cat(outs, dim=1)
        return out


class PSPHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, bins: list, n_classes: int):
        super().__init__()

        self.pp = PyramidPooling(in_channels, mid_channels, bins)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels+len(bins)*mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.drop = nn.Dropout2d(0.1)
        self.cls_top = nn.Conv2d(mid_channels, n_classes, kernel_size=1)

        self._init_weights()

    def forward(self, x: torch.Tensor):
        x = self.pp(x)
        x = self.conv(x)
        x = self.drop(x)
        out = self.cls_top(x)
        return out

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    if 'cls_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


class PSPAuxHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, n_classes: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.drop = nn.Dropout2d(0.1)
        self.cls_top = nn.Conv2d(mid_channels, n_classes, kernel_size=1)

        self._init_weights()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.drop(x)
        out = self.cls_top(x)
        return out

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    if 'cls_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
