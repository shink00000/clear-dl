import torch
import torch.nn as nn
import torch.nn.functional as F


class PoolingBlock(nn.Sequential):
    def __init__(self, pool_size: int, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(output_size=pool_size),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class OutBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, outs: tuple):
        out = torch.cat(outs, dim=1)
        return super().forward(out)


class PyramidPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bins: list):
        super().__init__()
        self.bins = bins
        for bin in self.bins:
            setattr(self, f'pc{bin}', PoolingBlock(bin, in_channels, out_channels))
        self.out_conv = OutBlock(in_channels+len(self.bins)*out_channels, out_channels)

        self._init_weights()

    def forward(self, x):
        _, _, H, W = x.size()
        outs = [x]
        for bin in self.bins:
            p = getattr(self, f'pc{bin}')(x)
            p = F.interpolate(p, size=(H, W), mode='bilinear', align_corners=True)
            outs.append(p)
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
