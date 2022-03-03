import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone
from ..necks.bifpn import BiFPN
from ..heads.efficient_head import EfficientSegHead
from ..losses import build_loss


class EfficientSeg(nn.Module):
    def __init__(self, size: str, backbone: dict, n_classes: int, feat_levels: list, output_size: list,
                 criterion: dict):
        super().__init__()

        # layers
        backbone_size, channels, n_bifpn_blocks, _ = {
            'd0': ('b0', 64, 3, 3),
            'd1': ('b1', 88, 4, 3),
            'd2': ('b2', 112, 5, 3),
            'd3': ('b3', 160, 6, 4),
            'd4': ('b4', 224, 7, 4),
            'd5': ('b5', 288, 7, 4),
            'd6': ('b6', 384, 8, 5),
            'd7': ('b7', 384, 8, 5)
        }[size]
        backbone.update({'size': backbone_size, 'feat_levels': feat_levels, 'out_channels': channels})
        self.backbone = build_backbone(backbone)
        self.neck = BiFPN(feat_levels, channels, n_bifpn_blocks)
        self.head = EfficientSegHead(feat_levels, channels, n_classes, output_size)

        # loss
        self.cls_loss = build_loss(criterion['cls_loss'])

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        outs = self.head(x)
        return outs

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cls_outs = outputs
        cls_loss = self.cls_loss(cls_outs, targets)
        return cls_loss

    def predict(self, outputs: torch.Tensor) -> torch.Tensor:
        cls_outs = outputs
        preds = F.softmax(cls_outs, dim=1)
        return preds
