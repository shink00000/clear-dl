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
