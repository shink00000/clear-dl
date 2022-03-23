import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone, get_channels
from ..losses import build_loss


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 groups: int = 1, act: str = 'relu'):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if act is not None:
            act_layer = {
                'relu': nn.ReLU(inplace=True),
                'sigmoid': nn.Sigmoid()
            }[act]
            modules.append(act_layer)
        super().__init__(*modules)


class SpacialPath(nn.Sequential):
    def __init__(self, out_channels: list):
        c_in = 3
        modules = []
        for i in range(len(out_channels)):
            c_out = out_channels[i]
            if i == 0:
                conv = ConvBlock(c_in, c_out, kernel_size=7, stride=2, padding=3)
            elif i < len(out_channels) - 1:
                conv = ConvBlock(c_in, c_out, kernel_size=3, stride=2, padding=1)
            else:
                conv = ConvBlock(c_in, c_out, kernel_size=1)
            modules.append(conv)
            c_in = c_out

        super().__init__(*modules)


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
        in_channels_4, in_channels_5 = get_channels(self.backbone, feat_levels)
        self.arm4 = ARM(in_channels_4, out_channels)
        self.arm5 = ARM(in_channels_5, out_channels)
        self.tail = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBlock(in_channels_5, out_channels, kernel_size=1)
        )
        self.up4 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.up5 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        feats = self.backbone(x)
        x4, x5 = (feats[level] for level in self.feat_levels)

        # up stream 1
        x = self.tail(x5)
        x = self.arm5(x5) + x
        out5 = self.up5(x)

        # up stream 2
        x = self.arm4(x4) + out5
        out4 = self.up4(x)

        return out4, out5


class FFM(nn.Module):
    """ Feature Fusion Module """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBlock(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_sp, x_cp):
        x = torch.cat([x_sp, x_cp], dim=1)
        x = self.conv(x)
        a = self.attn(x)
        out = x * a + x
        return out


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels: int, mid_channels: int, n_classes: int):
        super().__init__(
            ConvBlock(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_channels, n_classes, kernel_size=1)
        )


class AuxHead(nn.Module):
    def __init__(self, in_channels: list, mid_channels: list, n_classes: int,):
        assert len(in_channels) == len(mid_channels)
        super().__init__()
        for i in range(len(in_channels)):
            setattr(self, f'aux_{i}', nn.Sequential(
                ConvBlock(in_channels[i], mid_channels[i], kernel_size=3, padding=1),
                nn.Dropout2d(0.1),
                nn.Conv2d(mid_channels[i], n_classes, kernel_size=1)
            ))

    def forward(self, auxs: list):
        aux_outs = []
        for i in range(len(auxs)):
            aux_out = getattr(self, f'aux_{i}')(auxs[i])
            aux_outs.append(aux_out)
        return aux_outs


class BiSeNetV1(nn.Module):
    def __init__(self, spacial_path: dict, context_path: dict, ffm: dict, head: dict,
                 aux_head: dict, criterion: dict):
        super().__init__()

        self.spacial_path = SpacialPath(**spacial_path)
        self.context_path = ContextPath(**context_path)
        self.ffm = FFM(**ffm)
        self.head = SegmentationHead(**head)
        self.aux_head = AuxHead(**aux_head)

        self.cls_loss = build_loss(criterion['cls_loss'])
        self.aux_loss = build_loss(criterion['aux_loss'])

        self._init_weights()

    def forward(self, x):
        H, W = x.size()[2:]

        x_sp = self.spacial_path(x)
        x_cp4, x_cp5 = self.context_path(x)
        x = self.ffm(x_sp, x_cp4)
        out = self.head(x)
        aux1, aux2 = self.aux_head([x_cp4, x_cp5])

        # restore
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        aux1 = F.interpolate(aux1, size=(H, W), mode='bilinear', align_corners=True)
        aux2 = F.interpolate(aux2, size=(H, W), mode='bilinear', align_corners=True)

        return out, aux1, aux2

    def loss(self, outputs: tuple, targets: torch.Tensor) -> torch.Tensor:
        cls_outs, *aux_outs = outputs
        cls_loss = self.cls_loss(cls_outs, targets)
        aux_loss = 0
        for aux_out in aux_outs:
            aux_loss = self.aux_loss(aux_out, targets)
        return cls_loss + aux_loss

    def predict(self, outputs: tuple) -> torch.Tensor:
        cls_outs, *_ = outputs
        preds = F.softmax(cls_outs, dim=1)
        return preds

    def _init_weights(self):
        for name, m in self.named_modules():
            if 'backbone' in name:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
