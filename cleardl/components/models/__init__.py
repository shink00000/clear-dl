from .classifiers.resnet import ResNet
from .detectors.fcos import FCOS
from .detectors.retinanet import RetinaNet
from .detectors.efficientdet import EfficientDet
from .segmentors.unet import UNet

MODELS = {
    'ResNet': ResNet,
    'FCOS': FCOS,
    'RetinaNet': RetinaNet,
    'EfficientDet': EfficientDet,
    'UNet': UNet,
}
