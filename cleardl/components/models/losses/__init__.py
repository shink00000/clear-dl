import torch.nn as nn

from .focal_loss import FocalLoss
from .iou_loss import IoULossWithDistance

LOSSES = {
    'SmoothL1Loss': nn.SmoothL1Loss,
    'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'FocalLoss': FocalLoss,
    'IoULossWithDistance': IoULossWithDistance
}


def build_loss(loss: dict):
    return LOSSES[loss.pop('type')](**loss)
