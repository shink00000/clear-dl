from .classifiers.resnet import ResNet
from .detectors.fcos import FCOS
from .detectors.retinanet import RetinaNet
from .detectors.efficientdet import EfficientDet
from .segmentors.unet import UNet
from .segmentors.pspnet import PSPNet
from .segmentors.deeplab_v3plus import DeepLabV3Plus
from .segmentors.efficientseg import EfficientSeg

MODELS = {
    'ResNet': ResNet,
    'FCOS': FCOS,
    'RetinaNet': RetinaNet,
    'EfficientDet': EfficientDet,
    'UNet': UNet,
    'PSPNet': PSPNet,
    'DeepLabV3Plus': DeepLabV3Plus,
    'EfficientSeg': EfficientSeg
}
