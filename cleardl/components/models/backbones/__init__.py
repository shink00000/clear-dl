import torch
import torch.nn as nn

from .resnet import ResNet
from .resnext import ResNeXt
from .wide_resnet import WideResNet
from .efficientnet import EfficientNet

BACKBONES = {
    'ResNet': ResNet,
    'ResNeXt': ResNeXt,
    'WideResNet': WideResNet,
    'EfficientNet': EfficientNet
}


def build_backbone(backbone: dict):
    return BACKBONES[backbone.pop('type')](**backbone)


@torch.no_grad()
def get_channels(backbone: nn.Module, levels: list) -> list:
    x = torch.rand(2, 3, 128, 128)
    out = backbone(x)
    return [out[level].size(1) if level in out else -1 for level in levels]


@torch.no_grad()
def get_max_level(backbone: nn.Module) -> int:
    x = torch.rand(2, 3, 128, 128)
    out = backbone(x)
    return max(out)
