import torch.nn as nn

from .focal_loss import FocalLoss, SoftmaxFocalLoss
from .iou_loss import IoULossWithDistance
from .ohem_cross_entropy_loss import OHEMCrossEntropyLoss

LOSSES = {
    'SmoothL1Loss': nn.SmoothL1Loss,
    'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'FocalLoss': FocalLoss,
    'SoftmaxFocalLoss': SoftmaxFocalLoss,
    'IoULossWithDistance': IoULossWithDistance,
    'OHEMCrossEntropyLoss': OHEMCrossEntropyLoss
}


def build_loss(loss: dict):
    return LOSSES[loss.pop('type')](**loss)
