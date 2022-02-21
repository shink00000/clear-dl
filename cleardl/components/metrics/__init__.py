from .bbox_mean_ap import BBoxMeanAP
from .mean_iou import MeanIoU

METRICS = {
    'BBoxMeanAP': BBoxMeanAP,
    'MeanIoU': MeanIoU
}
