import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone
from ..necks.pyramid_pooling import PyramidPooling
from ..heads.psp_head import PSPHead
from ..losses import build_loss


class PSPNet(nn.Module):
    def __init__(self, feat_sizes: list, backbone: dict, n_classes: int, criterion: dict):
        super().__init__()
        assert len(feat_sizes) == 2

        backbone.update({'feat_sizes': feat_sizes, 'align_channel': False})
        self.backbone = build_backbone(backbone)

        self.aux_id, self.x_id = feat_sizes
        channels = self.backbone.get_channels()
        aux_channels, bout_channels = [channels[fsize] for fsize in feat_sizes]
        mid_channels = 512
        aux_mid_channels = 256
        self.neck = PyramidPooling(bout_channels, mid_channels)
        self.head = PSPHead(bout_channels+4*mid_channels, aux_channels, mid_channels, aux_mid_channels, n_classes)

        self.cls_loss = build_loss(criterion['cls_loss'])
        self.aux_loss = build_loss(criterion['aux_loss'])

    def forward(self, x):
        _, _, H, W = x.size()
        feats = self.backbone(x)
        x, aux = feats[self.x_id], feats[self.aux_id]
        x = self.neck(x)
        x, aux = self.head(x, aux)
        out = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        aux_out = F.interpolate(aux, size=(H, W), mode='bilinear', align_corners=True)
        return out, aux_out

    def loss(self, outputs: tuple, targets: torch.Tensor) -> torch.Tensor:
        cls_outs, aux_outs = outputs
        cls_loss = self.cls_loss(cls_outs, targets)
        aux_loss = self.aux_loss(aux_outs, targets)
        return cls_loss + 0.4 * aux_loss

    def predict(self, outputs: tuple) -> torch.Tensor:
        cls_outs, _ = outputs
        preds = F.softmax(cls_outs, dim=1)
        return preds
