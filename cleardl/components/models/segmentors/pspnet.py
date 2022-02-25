import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone
from ..necks.pyramid_pooling import PyramidPooling
from ..heads.psp_head import PSPHead
from ..heads.auxiliary_head import AuxiliaryHead
from ..losses import build_loss


class PSPNet(nn.Module):
    def __init__(self, feat_sizes: list, backbone: dict, bins: list, n_classes: int, criterion: dict):
        super().__init__()
        assert len(feat_sizes) == 2

        backbone.update({'feat_sizes': feat_sizes, 'align_channel': False})
        self.backbone = build_backbone(backbone)

        channels = self.backbone.get_channels()
        self.aux_id, self.x_id = feat_sizes
        aux_in_channels, in_channels = [channels[fsize] for fsize in feat_sizes]
        out_channels = 512
        self.neck = PyramidPooling(in_channels, out_channels, bins)
        self.head = PSPHead(out_channels, n_classes)
        self.aux_head = AuxiliaryHead(aux_in_channels, n_classes)

        self.cls_loss = build_loss(criterion['cls_loss'])
        self.aux_loss = build_loss(criterion['aux_loss'])

    def forward(self, x):
        _, _, H, W = x.size()
        feats = self.backbone(x)
        aux, x = feats[self.aux_id], feats[self.x_id]
        x = self.neck(x)
        x = self.head(x)
        out = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        # auxiliary
        aux = self.aux_head(aux)
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
