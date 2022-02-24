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


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.pc6 = PoolingBlock(6, in_channels, mid_channels)
        self.pc3 = PoolingBlock(3, in_channels, mid_channels)
        self.pc2 = PoolingBlock(2, in_channels, mid_channels)
        self.pc1 = PoolingBlock(1, in_channels, mid_channels)

        self._init_weights()

    def forward(self, x):
        _, _, H, W = x.size()
        p6 = F.interpolate(self.pc6(x), size=(H, W), mode='bilinear', align_corners=True)
        p3 = F.interpolate(self.pc3(x), size=(H, W), mode='bilinear', align_corners=True)
        p2 = F.interpolate(self.pc2(x), size=(H, W), mode='bilinear', align_corners=True)
        p1 = F.interpolate(self.pc1(x), size=(H, W), mode='bilinear', align_corners=True)

        out = torch.cat([x, p6, p3, p2, p1], dim=1)
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
