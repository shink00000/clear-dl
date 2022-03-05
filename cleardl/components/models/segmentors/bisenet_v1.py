import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone, get_channels
from ..losses import build_loss


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
    def __init__(self, out_channels: list):
        assert len(out_channels) == 3
        c0 = 3
        c1, c2, c3 = out_channels
        super().__init__(
            ConvBlock(c0, c1, kernel_size=7, stride=2, padding=3),
            ConvBlock(c1, c2, kernel_size=3, stride=2, padding=1),
            ConvBlock(c2, c3, kernel_size=3, stride=2, padding=1)
        )


class ARM(nn.Module):
    """ Attention Refinement Module """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBlock(out_channels, out_channels, kernel_size=1, act='sigmoid')
        )

    def forward(self, x):
        x = self.conv(x)
        a = self.attn(x)
        out = x * a
        return out


class ContextPath(nn.Module):
    def __init__(self, backbone: dict, feat_levels: list, out_channels: int):
        super().__init__()
        self.feat_levels = feat_levels

        # layers
        self.backbone = build_backbone(backbone)
        in_channels_3, in_channels_4, in_channels_5 = get_channels(self.backbone, feat_levels)
        self.arm1 = ARM(in_channels_4, out_channels)
        self.arm2 = ARM(in_channels_5, out_channels)
        self.tail = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels_5, out_channels, kernel_size=1)
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # for aux
        self.aux_in_channels_1 = in_channels_3
        self.aux_in_channels_2 = in_channels_4

    def forward(self, x):
        feats = self.backbone(x)
        x3, x4, x5 = (feats[level] for level in self.feat_levels)
        t = self.tail(x5)
        a = self.up(self.arm2(x5) + t)
        out = self.up(self.arm1(x4) + a)
        return out, x3, x4


class FFM(nn.Module):
    """ Feature Fusion Module """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_sp, x_cp):
        x_cp = F.interpolate(x_cp, size=x_sp.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_sp, x_cp], dim=1)
        x = self.conv(x)
        a = self.attn(x)
        out = x * a + x
        return out


class BiSeHead(nn.Module):
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

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.drop(x)
        x = self.cls_top(x)
        out = self.up(x)
        return out


class BiSeNetV1(nn.Module):
    def __init__(self, spacial_path: dict, context_path: dict, ffm: dict, head: dict,
                 aux_head: dict, criterion: dict):
        super().__init__()

        self.spacial_path = SpacialPath(**spacial_path)
        self.context_path = ContextPath(**context_path)
        self.ffm = FFM(**ffm)
        self.head = BiSeHead(**head)
        self.aux_head_1 = BiSeHead(in_channels=self.context_path.aux_in_channels_1, **aux_head)
        self.aux_head_2 = BiSeHead(in_channels=self.context_path.aux_in_channels_2, **aux_head)

        self.cls_loss = build_loss(criterion['cls_loss'])
        self.aux_loss = build_loss(criterion['aux_loss'])

        self._init_weights()

    def forward(self, x):
        x_sp = self.spacial_path(x)
        x_cp, aux_1, aux_2 = self.context_path(x)
        x = self.ffm(x_sp, x_cp)
        out = self.head(x)
        aux_out_1 = self.aux_head_1(aux_1)
        aux_out_2 = self.aux_head_2(aux_2)
        return out, aux_out_1, aux_out_2

    def loss(self, outputs: tuple, targets: torch.Tensor) -> torch.Tensor:
        cls_outs, *aux_outs = outputs
        cls_loss = self.cls_loss(cls_outs, targets)
        aux_loss = 0
        for aux_out in aux_outs:
            aux_loss = self.aux_loss(aux_out, targets)
        return cls_loss + 0.4 * aux_loss

    def predict(self, outputs: tuple) -> torch.Tensor:
        cls_outs, *_ = outputs
        preds = F.softmax(cls_outs, dim=1)
        return preds

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
