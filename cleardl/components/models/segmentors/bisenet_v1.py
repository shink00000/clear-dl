import torch
import torch.nn as nn

from ..backbones import build_backbone


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 act: str = 'relu'):
        act_layer = {
            'relu': nn.ReLU(inplace=True),
            'sigmoid': nn.Sigmoid()
        }[act]
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            act_layer
        )


class SpacialPath(nn.Sequential):
    def __init__(self, channel_list: list):
        assert len(channel_list) == 3
        c1, c2, c3 = channel_list
        super().__init__(
            ConvBlock(c1, c2, kernel_size=7, stride=2, padding=3),
            ConvBlock(c2, c2, kernel_size=3, stride=2, padding=1),
            ConvBlock(c2, c3, kernel_size=3, stride=2, padding=1)
        )


class ARM(nn.Module):
    """ Attention Refinement Module """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBlock(in_channels, out_channels, kernel_size=1, act='sigmoid')
        )

    def forward(self, x):
        y = x
        y = self.attn(y)
        out = x * y
        return out


class ContextPath(nn.Module):
    def __init__(self, backbone: dict, feat_levels: list):
        super().__init__()
        self.x1_id, self.x2_id = feat_levels
        backbone.update({'feat_levels': feat_levels, 'align_channel': False})
        self.backbone = build_backbone(backbone)
        channels = self.backbone.get_channels()
        c1, c2 = channels[self.x1_id], channels[self.x2_id]
        channels = 128
        self.arm1 = ARM(c1, channels)
        self.arm2 = ARM(c2, channels)
        self.tail = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feats = self.backbone(x)
        x1, x2 = feats[self.x1_id], feats[self.x2_id]
        t = self.tail(x2)
        t = self.arm2(x2) + t
        out = self.arm1(x1) + t
        return out


class FFM(nn.Module):
    """ Feature Fusion Module """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

    def forward(self, x_sp, x_cn):
        x = torch.cat([x_sp, x_cn], dim=1)
        x = self.conv(x)
        y = x
        y = self.attn(y)
        out = x * y + x
        return out


class BiSeNetV1(nn.Module):
    def __init__(self, feat_levels: list, backbone: dict):
        super().__init__()

        backbone.update({'feat_levels': feat_levels, 'align_channel': False})
        self.backbone = build_backbone(backbone)
