import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone, get_channels, get_max_level
from ..extras import build_extra
from ..necks.bifpn import BiFPN
from ..heads.efficient_head import EfficientSegHead
from ..losses import build_loss
from ..utils.replace_layer import replace_layer_


class EfficientSeg(nn.Module):
    def __init__(self, size: str, feat_levels: list, backbone: dict, extra: dict, neck: dict,
                 head: dict, criterion: dict, replace: dict = None):
        super().__init__()

        # definition by size
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
        backbone.update({'size': backbone_size})
        extra.update({
            'out_channels': channels
        })
        neck.update({
            'channels': channels,
            'n_blocks': n_bifpn_blocks
        })
        head.update({
            'in_channels': channels
        })

        # layers
        self.backbone = build_backbone(backbone)
        extra.update({
            'in_channels': get_channels(self.backbone, feat_levels),
            'max_level': get_max_level(self.backbone)
        })
        self.extra = build_extra(extra)
        self.neck = BiFPN(**neck)
        self.head = EfficientSegHead(**head)

        if replace is not None:
            replace_layer_(self, **replace)

        # loss
        self.cls_loss = build_loss(criterion['cls_loss'])

    def forward(self, x):
        H, W = x.size()[2:]

        x = self.backbone(x)
        x = self.extra(x)
        x = self.neck(x)
        outs = self.head(x)

        # restore
        outs = F.interpolate(outs, size=(H, W), mode='bilinear', align_corners=True)

        return outs

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cls_outs = outputs
        cls_loss = self.cls_loss(cls_outs, targets)
        return cls_loss

    def predict(self, outputs: torch.Tensor) -> torch.Tensor:
        cls_outs = outputs
        preds = F.softmax(cls_outs, dim=1)
        return preds
