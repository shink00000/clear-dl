import torch
import torch.nn as nn
from itertools import product

from ..backbones import build_backbone
from ..necks.fpn import FPN
from ..heads.fcos_head import FCOSHead
from ..losses import build_loss
from ..postprocesses import build_postprocess


class FCOS(nn.Module):
    def __init__(self, backbone: dict, n_classes: int, input_size: list, feat_sizes: list,
                 criterion: dict, postprocess: dict):
        super().__init__()

        # layers
        channels = 256
        backbone.update({'feat_sizes': feat_sizes, 'out_channels': channels})
        self.backbone = build_backbone(backbone)
        self.neck = FPN(feat_sizes, channels)
        self.head = FCOSHead(feat_sizes, channels, n_classes)

        # property
        H, W = input_size
        strides = [2**i for i in feat_sizes]
        all_points = []
        for stride in strides:
            points = [[x, y] for y, x in product(
                range(stride//2, H, stride), range(stride//2, W, stride)
            )]
            all_points.extend(points)
        self.all_points = nn.Parameter(torch.Tensor(all_points), requires_grad=False)

        # loss
        self.reg_loss = build_loss(criterion['reg_loss'])
        self.cls_loss = build_loss(criterion['cls_loss'])
        self.cnt_loss = build_loss(criterion['cnt_loss'])

        # postprocess
        self.postprocess = build_postprocess(postprocess)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        outs = self.head(x)
        return outs

    def loss(self, outputs: tuple, targets: tuple) -> torch.Tensor:
        reg_outs, cls_outs, cnt_outs = outputs
        reg_targets, cls_targets, cnt_targets = targets

        pos_mask = cls_targets > 0
        neg_mask = cls_targets == 0
        N = pos_mask.sum()

        if N > 0:
            reg_loss = self.reg_loss(reg_outs[pos_mask], reg_targets[pos_mask]) / N
            cnt_loss = self.cnt_loss(cnt_outs[pos_mask], cnt_targets[pos_mask]) / N
            cls_loss = self.cls_loss(cls_outs[pos_mask + neg_mask], cls_targets[pos_mask + neg_mask]) / N
            loss = reg_loss + cnt_loss + cls_loss
        else:
            loss = self.cls_loss(cls_outs[neg_mask], cls_targets[neg_mask])

        return loss

    def predict(self, outs: tuple) -> tuple:
        reg_outs, cls_outs, cnt_outs = outs
        bboxes = self.distance2bbox(reg_outs)
        scores = cls_outs.sigmoid() * cnt_outs.sigmoid()
        bboxes, scores, class_ids = self.postprocess(bboxes, scores)
        return bboxes, scores, class_ids

    def distance2bbox(self, reg_outs) -> torch.Tensor:
        xmin = self.all_points[..., 0] - reg_outs[..., 0]
        ymin = self.all_points[..., 1] - reg_outs[..., 1]
        xmax = self.all_points[..., 0] + reg_outs[..., 2]
        ymax = self.all_points[..., 1] + reg_outs[..., 3]

        return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


class FCOSEncoder(nn.Module):
    def __init__(self, input_size: list, feat_sizes: list):
        super().__init__()
        self.INF = 1e8
        H, W = input_size
        strides = [2**i for i in feat_sizes]
        regress_ranges = [[s*4, s*8] for s in strides]
        regress_ranges[0][0], regress_ranges[-1][-1] = -1, self.INF

        all_points = []
        all_regress_ranges = []
        for stride, regress_range in zip(strides, regress_ranges):
            points = [[x, y] for y, x in product(
                range(stride//2, H, stride), range(stride//2, W, stride)
            )]
            all_points.extend(points)
            all_regress_ranges.extend([regress_range for _ in range(len(points))])

        self.all_points = torch.Tensor(all_points)
        self.all_regress_ranges = torch.Tensor(all_regress_ranges)

    def forward(self, bboxes: torch.Tensor, labels: torch.Tensor) -> tuple:
        xs, ys = self.all_points.split(1, dim=1)
        ls, us = self.all_regress_ranges.split(1, dim=1)

        left = xs - bboxes[..., 0]
        top = ys - bboxes[..., 1]
        right = bboxes[..., 2] - xs
        bottom = bboxes[..., 3] - ys

        distances = torch.stack([left, top, right, bottom], dim=-1)
        areas = (distances[..., 0] + distances[..., 2]) * (distances[..., 1] + distances[..., 3])

        inside_bboxes = distances.min(dim=-1)[0] > 0
        inside_ranges = torch.logical_and(ls <= distances.max(dim=-1)[0], distances.max(dim=-1)[0] <= us)
        areas[inside_bboxes == 0] = self.INF
        areas[inside_ranges == 0] = self.INF
        min_area, min_area_inds = areas.min(dim=1)

        cls_targets = labels[min_area_inds]
        cls_targets[min_area == self.INF] = 0
        reg_targets = distances[range(len(min_area_inds)), min_area_inds]
        cnt_targets = torch.div(
            torch.minimum(reg_targets[..., [0, 2]], reg_targets[..., [1, 3]]),
            torch.maximum(reg_targets[..., [0, 2]], reg_targets[..., [1, 3]])
        ).prod(dim=-1, keepdim=True).sqrt()

        return reg_targets, cls_targets, cnt_targets
