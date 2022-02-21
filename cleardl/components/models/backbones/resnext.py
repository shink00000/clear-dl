from torchvision.models import (
    resnext50_32x4d,
    resnext101_32x8d
)

from .resnet import ResNet


class ResNeXt(ResNet):

    def _build_base(self, depth: int):
        return {
            50: resnext50_32x4d,
            101: resnext101_32x8d,
        }[depth]
