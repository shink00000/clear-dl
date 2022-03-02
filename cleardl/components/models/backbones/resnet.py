import torch
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101
)

from .basebackbone import BaseBackBone


class ResNet(BaseBackBone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, max_size=5, **kwargs)

    def _build(self, depth: int, weights: str = 'default', frozen_stages: int = -1, **kwargs):
        if weights == 'default':
            base = self._build_base(depth)(pretrained=True, **kwargs)
        else:
            base = self._build_base(depth)(pretrained=False, **kwargs)
            if weights is not None:
                base.load_state_dict(torch.load(weights, map_location='cpu'))

        freeze = frozen_stages >= 0
        for name, m in base.named_children():
            if 'layer' in name and int(name[-1]) > frozen_stages:
                freeze = False
            if freeze:
                m.requires_grad_(False)
            if name == 'avgpool':
                break
            else:
                setattr(self, name, m)

    def _build_base(self, depth: int):
        return {
            18: resnet18,
            34: resnet34,
            50: resnet50,
            101: resnet101,
        }[depth]

    def _forward(self, x: torch.Tensor) -> dict:
        feats = {}
        feats[1] = self.relu(self.bn1(self.conv1(x)))
        feats[2] = self.layer1(self.maxpool(feats[1]))
        feats[3] = self.layer2(feats[2])
        feats[4] = self.layer3(feats[3])
        feats[5] = self.layer4(feats[4])
        return feats
