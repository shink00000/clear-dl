from torchvision.models import (
    wide_resnet50_2,
    wide_resnet101_2
)

from .resnet import ResNet


class WideResNet(ResNet):

    def _build_base(self, depth: int):
        return {
            50: wide_resnet50_2,
            101: wide_resnet101_2,
        }[depth]
