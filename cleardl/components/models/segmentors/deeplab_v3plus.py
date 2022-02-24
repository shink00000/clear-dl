import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone
from ..necks.aspp import ASPP
from ..heads.deeplab_v3plus_head import DeepLabV3PlusHead
from ..losses import build_loss


class DeepLabV3Plus(nn.Module):
    def __init__(self, feat_sizes: list, backbone: dict, rates: list, n_classes: int, criterion: dict):
        super().__init__()
        assert len(feat_sizes) == 2

        backbone.update({'feat_sizes': feat_sizes, 'align_channel': False})
        self.backbone = build_backbone(backbone)

        channels = self.backbone.get_channels()
        self.low_id, self.x_id = feat_sizes
        low_channels, bout_channels = [channels[fsize] for fsize in feat_sizes]
        channels = 256
        self.neck = ASPP(bout_channels, channels, rates)
        self.head = DeepLabV3PlusHead(channels, low_channels, n_classes)

        self.cls_loss = build_loss(criterion['cls_loss'])

    def forward(self, x):
        _, _, H, W = x.size()
        feats = self.backbone(x)
        x, low = feats[self.x_id], feats[self.low_id]
        x = self.neck(x)
        x = self.head(x, low)
        out = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return out

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cls_loss = self.cls_loss(outputs, targets)
        return cls_loss

    def predict(self, outputs: tuple) -> torch.Tensor:
        preds = F.softmax(outputs, dim=1)
        return preds
