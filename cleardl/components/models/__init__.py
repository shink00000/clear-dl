from .detectors.fcos import FCOS
from .detectors.retinanet import RetinaNet
from .detectors.efficientdet import EfficientDet
from .segmentors.unet import UNet
from .segmentors.pspnet import PSPNet
from .segmentors.deeplab_v3p import DeepLabV3P
from .segmentors.efficientseg import EfficientSeg
from .segmentors.bisenet_v1 import BiSeNetV1

MODELS = {
    'FCOS': FCOS,
    'RetinaNet': RetinaNet,
    'EfficientDet': EfficientDet,
    'UNet': UNet,
    'PSPNet': PSPNet,
    'DeepLabV3P': DeepLabV3P,
    'EfficientSeg': EfficientSeg,
    'BiSeNetV1': BiSeNetV1,
}
