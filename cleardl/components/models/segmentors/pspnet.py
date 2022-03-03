import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone
from ..necks.pyramid_pooling import PyramidPooling
from ..heads.psp_head import PSPHead
from ..heads.auxiliary_head import AuxiliaryHead
from ..losses import build_loss


class PSPNet(nn.Module):
    def __init__(self, backbone: dict, bins: list, n_classes: int, output_size: list, criterion: dict):
        super().__init__()

        backbone.update({'feat_levels': [4, 5], 'align_channel': False})
        self.backbone = build_backbone(backbone)

        channels = self.backbone.get_channels()
        aux_in_channels, in_channels = channels[4], channels[5]
        out_channels = 512
        self.neck = PyramidPooling(in_channels, out_channels, bins)
        self.head = PSPHead(out_channels, n_classes, output_size)
        self.aux_head = AuxiliaryHead(aux_in_channels, n_classes, output_size)

        self.cls_loss = build_loss(criterion['cls_loss'])
        self.aux_loss = build_loss(criterion['aux_loss'])

    def forward(self, x):
        feats = self.backbone(x)
        aux, x = feats[4], feats[5]
        x = self.neck(x)
        out = self.head(x)
        aux_out = self.aux_head(aux)

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
