import torch
from torchvision.models import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
)

from .basebackbone import BaseBackBone


class EfficientNet(BaseBackBone):

    def __init__(self, *args, **kwargs):
        self.pickup_id_to_fsize = {
            1: 1,
            2: 2,
            3: 3,
            5: 4,
            8: 5
        }
        super().__init__(*args, max_size=5, **kwargs)

    def _build(self, size: str, weights: str = 'default', frozen_stages: int = -1, **kwargs):
        if weights == 'default':
            base = self._build_base(size)(pretrained=True, **kwargs)
        else:
            base = self._build_base(size)(pretrained=False, **kwargs)
            if weights is not None:
                base.load_state_dict(torch.load(weights, map_location='cpu'))

        self.features = base.features
        for i, m in enumerate(self.features):
            if i <= frozen_stages:
                m.requires_grad_(False)
            else:
                break

    def _build_base(self, size: str):
        return {
            'b0': efficientnet_b0,
            'b1': efficientnet_b1,
            'b2': efficientnet_b2,
            'b3': efficientnet_b3,
            'b4': efficientnet_b4,
            'b5': efficientnet_b5,
            'b6': efficientnet_b6,
            'b7': efficientnet_b7,
        }[size]

    def _forward(self, x: torch.Tensor) -> dict:
        feats = {}
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.pickup_id_to_fsize:
                feats[self.pickup_id_to_fsize[i]] = x
        return feats
