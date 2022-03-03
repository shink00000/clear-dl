import torch
import torch.nn as nn
from itertools import product
from torchvision.ops import box_iou, box_convert

from ..backbones import build_backbone
from ..necks.fpn import FPN
from ..heads.retina_head import RetinaHead
from ..losses import build_loss
from ..postprocesses import build_postprocess


class RetinaNet(nn.Module):
    def __init__(self, backbone: dict, n_classes: int, input_size: list, feat_levels: list,
                 criterion: dict, postprocess: dict):
        super().__init__()

        # layers
        channels = 256
        backbone.update({'feat_levels': feat_levels, 'out_channels': channels})
        self.backbone = build_backbone(backbone)
        self.neck = FPN(feat_levels, channels)
        self.head = RetinaHead(feat_levels, channels, n_classes)

        # property
        H, W = input_size
        strides = [2**i for i in feat_levels]
        prior_boxes = []
        for stride in strides:
            for cy, cx in product(range(stride//2, H, stride), range(stride//2, W, stride)):
                for aspect in [0.5, 1.0, 2.0]:
                    for scale in [0, 1/3, 2/3]:
                        h = 4 * stride * pow(2, scale) * pow(aspect, 1/2)
                        w = 4 * stride * pow(2, scale) * pow(1/aspect, 1/2)
                        prior_boxes.append([cx, cy, w, h])
        self.prior_boxes = nn.Parameter(torch.Tensor(prior_boxes), requires_grad=False)

        # loss
        self.reg_loss = build_loss(criterion['reg_loss'])
        self.cls_loss = build_loss(criterion['cls_loss'])

        # postprocess
        self.postprocess = build_postprocess(postprocess)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        outs = self.head(x)
        return outs

    def loss(self, outputs: tuple, targets: tuple) -> torch.Tensor:
        reg_outs, cls_outs = outputs
        reg_targets, cls_targets = targets

        pos_mask = cls_targets > 0
        neg_mask = cls_targets == 0
        N = pos_mask.sum()

        if N > 0:
            reg_loss = self.reg_loss(reg_outs[pos_mask], reg_targets[pos_mask]) / N
            cls_loss = self.cls_loss(cls_outs[pos_mask + neg_mask], cls_targets[pos_mask + neg_mask]) / N
            loss = reg_loss + cls_loss
        else:
            loss = self.cls_loss(cls_outs[neg_mask], cls_targets[neg_mask])

        return loss

    def predict(self, outputs: tuple) -> tuple:
        reg_outs, cls_outs = outputs
        bboxes = box_convert(self.delta2bbox(reg_outs), 'cxcywh', 'xyxy')
        scores = cls_outs.sigmoid()
        bboxes, scores, class_ids = self.postprocess(bboxes, scores)
        return bboxes, scores, class_ids

    def delta2bbox(self, reg_outs) -> torch.Tensor:
        cx = self.prior_boxes[..., 0] + 0.1 * reg_outs[..., 0] * self.prior_boxes[..., 2]
        cy = self.prior_boxes[..., 1] + 0.1 * reg_outs[..., 1] * self.prior_boxes[..., 3]
        w = self.prior_boxes[..., 2] * (0.2 * reg_outs[..., 2]).exp()
        h = self.prior_boxes[..., 3] * (0.2 * reg_outs[..., 3]).exp()

        return torch.stack([cx, cy, w, h], dim=-1)


class RetinaEncoder(nn.Module):
    def __init__(self, input_size: list, feat_levels: list, iou_threshs: tuple = (0.4, 0.5)):
        super().__init__()
        H, W = input_size
        strides = [2**i for i in feat_levels]
        prior_boxes = []
        for stride in strides:
            for cy, cx in product(range(stride//2, H, stride), range(stride//2, W, stride)):
                for aspect in [0.5, 1.0, 2.0]:
                    for scale in [0, 1/3, 2/3]:
                        h = 4 * stride * pow(2, scale) * pow(aspect, 1/2)
                        w = 4 * stride * pow(2, scale) * pow(1/aspect, 1/2)
                        prior_boxes.append([cx, cy, w, h])
        self.prior_boxes = torch.Tensor(prior_boxes)
        self.neg_thresh, self.pos_thresh = iou_threshs

    def forward(self, bboxes: torch.Tensor, labels: torch.Tensor) -> tuple:
        ious = box_iou(bboxes, box_convert(self.prior_boxes, 'cxcywh', 'xyxy'))
        max_ious, match_ids = ious.max(dim=0)

        # force assign (all bboxes match one or more prior boxes)
        force_assign_ids = ious.argmax(dim=1)
        match_ids[force_assign_ids] = torch.arange(len(bboxes))
        max_ious[force_assign_ids] = self.pos_thresh

        reg_targets = self.bbox2delta(box_convert(bboxes, 'xyxy', 'cxcywh')[match_ids])
        cls_targets = labels[match_ids]
        cls_targets[max_ious < self.pos_thresh] = -1
        cls_targets[max_ious < self.neg_thresh] = 0

        return reg_targets, cls_targets

    def bbox2delta(self, bboxes) -> torch.Tensor:
        dcx = (bboxes[..., 0] - self.prior_boxes[..., 0]) / self.prior_boxes[..., 2] / 0.1
        dcy = (bboxes[..., 1] - self.prior_boxes[..., 1]) / self.prior_boxes[..., 3] / 0.1
        dw = (bboxes[..., 2] / self.prior_boxes[..., 2]).log() / 0.2
        dh = (bboxes[..., 3] / self.prior_boxes[..., 3]).log() / 0.2

        return torch.stack([dcx, dcy, dw, dh], dim=-1)
