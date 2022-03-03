import torch
import torch.nn as nn
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


class EfficientNet(nn.Module):
    def __init__(self, size: str, weights: str = 'default', frozen_stages: int = -1, **kwargs):
        super().__init__()

        if weights == 'default':
            base = self._models(size)(pretrained=True, **kwargs)
        else:
            base = self._models(size)(pretrained=False, **kwargs)
            if weights is not None:
                base.load_state_dict(torch.load(weights, map_location='cpu'))

        self.features = base.features
        for i, m in enumerate(self.features):
            if i <= frozen_stages:
                m.requires_grad_(False)
            else:
                break

    def forward(self, x: torch.Tensor) -> dict:
        id2level = {1: 1, 2: 2, 3: 3, 5: 4, 8: 5}
        feats = {}
        for i, m in enumerate(self.features):
            x = m(x)
            if i in id2level:
                feats[id2level[i]] = x
        return feats

    def _models(self, size: str):
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
