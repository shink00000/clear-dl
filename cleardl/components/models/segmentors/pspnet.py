import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone, get_channels
from ..heads.psp_head import PSPHead, PSPAuxHead
from ..losses import build_loss
from ..utils.replace_layer import replace_bn_to_wsgn_


class PSPNet(nn.Module):
    def __init__(self, feat_levels: list, backbone: dict, head: dict, aux_head: dict,
                 criterion: dict, replace_bn_to_wsgn: bool = True):
        super().__init__()

        assert len(feat_levels) == 2
        self.feat_levels = feat_levels

        # layers
        self.backbone = build_backbone(backbone)
        aux_in_channels, in_channels = get_channels(self.backbone, feat_levels)
        head.update({'in_channels': in_channels})
        aux_head.update({'in_channels': aux_in_channels})
        self.head = PSPHead(**head)
        self.aux_head = PSPAuxHead(**aux_head)

        if replace_bn_to_wsgn:
            replace_bn_to_wsgn_(self)

        self.cls_loss = build_loss(criterion['cls_loss'])
        self.aux_loss = build_loss(criterion['aux_loss'])

    def forward(self, x):
        feats = self.backbone(x)
        aux, x = (feats[level] for level in self.feat_levels)
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
