import torch
import torch.nn as nn
import torch.nn.functional as F

from .bisenet_v1 import ConvBlock
from ..losses import build_loss


class DetailBranch(nn.Sequential):
    def __init__(self, out_channels: list):
        assert len(out_channels) == 3
        c0 = 3
        c1, c2, c3 = out_channels
        super().__init__(
            nn.Sequential(
                ConvBlock(c0, c1, kernel_size=3, stride=2, padding=1),
                ConvBlock(c1, c1, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                ConvBlock(c1, c2, kernel_size=3, stride=2, padding=1),
                ConvBlock(c1, c1, kernel_size=3, padding=1),
                ConvBlock(c1, c1, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                ConvBlock(c2, c3, kernel_size=3, stride=2, padding=1),
                ConvBlock(c3, c3, kernel_size=3, padding=1),
                ConvBlock(c3, c3, kernel_size=3, padding=1),
            )
        )


class StemBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            ConvBlock(out_channels, out_channels, kernel_size=1),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )
        self.pool = nn.MaxPool2d(in_channels, stride=2, padding=1)
        self.out_c = ConvBlock(2*out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        out = self.out_c(x)
        return out


class GELayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        assert stride in (1, 2)
        if stride == 1:
            self.main_path = nn.Sequential(
                ConvBlock(in_channels, 6*in_channels, kernel_size=3, padding=1),
                ConvBlock(6*in_channels, 6*in_channels, kernel_size=3, padding=1, groups=6*in_channels),
                ConvBlock(6*in_channels, out_channels, kernel_size=1, act=None)
            )
            self.shortcut = nn.Identity()
        else:
            self.main_path = nn.Sequential(
                ConvBlock(in_channels, 6*in_channels, kernel_size=3, padding=1),
                ConvBlock(6*in_channels, 6*in_channels, kernel_size=3, stride=2, padding=1, groups=6*in_channels),
                ConvBlock(6*in_channels, 6*in_channels, kernel_size=3, padding=1, groups=6*in_channels),
                ConvBlock(6*in_channels, out_channels, kernel_size=1, act=None)
            )
            self.shortcut = nn.Sequential(
                ConvBlock(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels),
                ConvBlock(in_channels, out_channels, kernel_size=1, act=None)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.main_path(x)
        x2 = self.shortcut(x)
        out = self.relu(x1 + x2)
        return out


class ContextEmbeddingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, kernel_size=1)
        self.out_c = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv(self.bn(self.pool(x)))
        out = self.out_c(x1 + x)
        return out


class SemanticBranch(nn.Module):
    def __init__(self, out_channels: list):
        assert len(out_channels) == 4
        c0 = 3
        c2, c3, c4, c5 = out_channels
        super().__init__()
        self.blocks = nn.ModuleList([
            StemBlock(c0, c2),
            nn.Sequential(
                GELayer(c2, c3, stride=2),
                GELayer(c3, c3, stride=1)
            ),
            nn.Sequential(
                GELayer(c3, c4, stride=2),
                GELayer(c4, c4, stride=1)
            ),
            nn.Sequential(
                GELayer(c4, c5, stride=2),
                GELayer(c5, c5, stride=1),
                GELayer(c5, c5, stride=1),
                GELayer(c5, c5, stride=1)
            ),
            ContextEmbeddingBlock(c5, c5)
        ])

    def forward(self, x):
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        outs[-1] = x  # remove CEBlock output
        return outs


class BilateralGuidedAggregationLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.detail_dconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self.detail_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Sigmoid()
        )
        self.semantic_dconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        self.out_c = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x_det, x_sem):
        x_det_dc = self.detail_dconv(x_det)
        x_det_down = self.detail_conv(x_det)
        x_sem_dc = self.semantic_dconv(x_sem)
        x_sem_up = self.semantic_conv(x_sem)
        x = x_det_dc * x_sem_up + self.up(x_sem_dc * x_det_down)
        out = self.out_c(x)
        return out


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels: int, mid_channels: int, n_classes: int, input_size: list):
        super().__init__(
            ConvBlock(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels, n_classes, kernel_size=1),
            nn.UpsamplingBilinear2d(input_size)
        )


class AuxHead(nn.Module):
    def __init__(self, in_channels: list, mid_channels: list, n_classes: int, input_size: list):
        assert len(in_channels) == len(mid_channels)
        super().__init__()
        self.heads = nn.ModuleList([
            SegmentationHead(in_channels[i], mid_channels[i], n_classes, input_size)
            for i in range(len(in_channels))
        ])

    def forward(self, auxs: list):
        aux_outs = []
        for i in range(len(auxs)):
            aux_out = self.heads[i](auxs[i])
            aux_outs.append(aux_out)
        return aux_out


class BiSeNetV2(nn.Module):
    def __init__(self, detail_branch: dict, semantic_branch: dict, bga: dict,
                 head: dict, aux_head: dict, criterion: dict):
        super().__init__()
        self.detail_branch = DetailBranch(**detail_branch)
        self.semantic_branch = SemanticBranch(**semantic_branch)
        self.bga = BilateralGuidedAggregationLayer(**bga)
        self.head = SegmentationHead(**head)
        self.aux_head = AuxHead(**aux_head)

        self.cls_loss = build_loss(criterion['cls_loss'])
        self.aux_loss = build_loss(criterion['aux_loss'])

        self._init_weights()

    def forward(self, x):
        x_det = self.detail_branch(x)
        *auxs, x_sem = self.semantic_branch(x)
        x = self.bga(x_det, x_sem)
        out = self.head(x)
        aux_outs = self.aux_head(auxs)
        return out, aux_outs

    def loss(self, outputs: tuple, targets: torch.Tensor) -> torch.Tensor:
        cls_out, *aux_outs = outputs
        cls_loss = self.cls_loss(cls_out, targets)
        aux_loss = 0
        for aux_out in aux_outs:
            aux_loss = self.aux_loss(aux_out, targets)
        return cls_loss + aux_loss

    def predict(self, outputs: tuple) -> torch.Tensor:
        cls_out, *_ = outputs
        preds = F.softmax(cls_out, dim=1)
        return preds

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
