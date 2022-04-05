import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone, get_channels
from ..heads.deeplab_v3p_head import DeepLabV3PHead, DeepLabV3PAuxHead
from ..losses import build_loss
from ..utils.replace_layer import replace_layer_


class DeepLabV3P(nn.Module):
    def __init__(self, feat_levels: list, backbone: dict, head: dict, aux_head: dict,
                 criterion: dict, replace: dict = None):
        super().__init__()

        assert len(feat_levels) == 3
        self.feat_levels = feat_levels

        # layers
        self.backbone = build_backbone(backbone)
        low_in_channels, aux_in_channels, in_channels = get_channels(self.backbone, feat_levels)
        head.update({'in_channels': in_channels, 'low_in_channels': low_in_channels})
        aux_head.update({'in_channels': aux_in_channels})
        self.head = DeepLabV3PHead(**head)
        self.aux_head = DeepLabV3PAuxHead(**aux_head)

        self.cls_loss = build_loss(criterion['cls_loss'])
        self.aux_loss = build_loss(criterion['aux_loss'])

        if replace is not None:
            replace_layer_(self, **replace)

    def forward(self, x):
        H, W = x.size()[2:]

        feats = self.backbone(x)
        low, aux, x = (feats[level] for level in self.feat_levels)
        out = self.head(x, low)
        aux_out = self.aux_head(aux)

        # restore
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        aux_out = F.interpolate(aux_out, size=(H, W), mode='bilinear', align_corners=True)

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
