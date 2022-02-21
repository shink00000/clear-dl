from torchvision.transforms import Compose

from .color import PhotoMetricDistortion, Normalize
from .geometry import RandomExpand, RandomMinIoUCrop, RandomHorizontalFlip, Resize, RandomMinAreaCrop
from .dropout import PixelwiseCutOff

from ...models.detectors.fcos import FCOSEncoder
from ...models.detectors.retinanet import RetinaEncoder
from ...models.detectors.efficientdet import EfficientEncoder

TRANSFORMS = {
    'PhotoMetricDistortion': PhotoMetricDistortion,
    'Normalize': Normalize,
    'RandomExpand': RandomExpand,
    'RandomMinIoUCrop': RandomMinIoUCrop,
    'RandomMinAreaCrop': RandomMinAreaCrop,
    'RandomHorizontalFlip': RandomHorizontalFlip,
    'Resize': Resize,
    'PixelwiseCutOff': PixelwiseCutOff
}

ENCODERS = {
    'FCOSEncoder': FCOSEncoder,
    'RetinaEncoder': RetinaEncoder,
    'EfficientEncoder': EfficientEncoder
}


def build_transforms(cfgs: list):
    return Compose([TRANSFORMS[c.pop('type')](**c) for c in cfgs])


def build_encoder(cfg: dict):
    return ENCODERS[cfg.pop('type')](**cfg)
