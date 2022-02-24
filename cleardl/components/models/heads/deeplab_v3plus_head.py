import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..layers.separatable_conv import SeparatableConv2d


class LowLevelProjection(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Branch(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            SeparatableConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, in_channels: int, low_in_channels: int, n_classes: int):
        super().__init__()
        low_out_channels = 48
        self.low_proj = LowLevelProjection(low_in_channels, low_out_channels)
        self.cls_branch = Branch(in_channels+low_out_channels, in_channels)
        self.cls_top = nn.Conv2d(in_channels, n_classes, kernel_size=1)

        self._init_weights()

    def forward(self, x: torch.Tensor, low: torch.Tensor):
        low = self.low_proj(low)
        x = F.interpolate(x, size=low.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, low], dim=1)
        cls_outs = self.cls_top(self.cls_branch(x))
        return cls_outs

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    if '_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
