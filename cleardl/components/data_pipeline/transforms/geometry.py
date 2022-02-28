import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import random


class RandomExpand(nn.Module):
    def __init__(self, ratio_range: list = [1.0, 4.0], p: float = 0.5):
        super().__init__()
        self.min_r, self.max_r = ratio_range
        self.p = p

    def forward(self, dc: dict) -> dict:
        if random.random() < self.p:
            _, h, w = dc['image'].shape
            pad_lengths = [int(e * (self.max_r - self.min_r) * random.random() / 2) for e in [w, w, h, h]]
            dc['image'] = F.pad(dc['image'], pad_lengths)
            if 'bboxes' in dc:
                dc['bboxes'][:, [0, 2]] += pad_lengths[0]
                dc['bboxes'][:, [1, 3]] += pad_lengths[2]
            if 'masks' in dc:
                dc['masks'] = F.pad(dc['masks'], pad_lengths)
            if 'label' in dc and dc['label'].ndim == 2:
                dc['label'] = F.pad(dc['label'], pad_lengths)
        return dc


class RandomMinIoUCrop(nn.Module):
    def __init__(self, min_ious: list = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
                 n_cands: int = 50, aspect_range: list = [0.5, 2.0], p: float = 0.5):
        super().__init__()
        self.min_ious = min_ious
        self.n_cands = n_cands
        self.aspect_range = aspect_range
        self.p = p

    def forward(self, dc: dict) -> dict:
        if random.random() < self.p and ('bboxes' in dc):
            iou_thresh = random.choice(self.min_ious)

            # create candidate crop regions
            _, h, w = dc['image'].shape
            wh = torch.Tensor([w, h])
            whcrop = (wh * torch.empty(self.n_cands, 2).uniform_(0.3, 1)).int()
            xymin = ((wh - whcrop) * torch.rand(self.n_cands, 2)).int()
            xymax = xymin + whcrop
            crop_regions = torch.cat([xymin, xymax], dim=1)

            # filter by conditions
            aspect_ratio = whcrop[:, 0] / whcrop[:, 1]
            crop_regions = crop_regions[aspect_ratio.clip(*self.aspect_range) == aspect_ratio]
            min_ious = self._calc_iou(crop_regions, dc['bboxes']).min(dim=1)[0]
            crop_regions = crop_regions[min_ious > iou_thresh]
            if len(crop_regions) > 0:
                l, t, r, b = crop_regions[0]
                dc['image'] = dc['image'][:, t:b, l:r]
                dc['bboxes'][:, [0, 2]] -= l
                dc['bboxes'][:, [1, 3]] -= t
                dc['bboxes'][:, [0, 2]] = dc['bboxes'][:, [0, 2]].clip(min=0, max=r-l)
                dc['bboxes'][:, [1, 3]] = dc['bboxes'][:, [1, 3]].clip(min=0, max=b-t)
                if 'masks' in dc:
                    dc['masks'] = dc['masks'][:, t:b, l:r]
                if 'label' in dc and dc['label'].ndim == 2:
                    dc['label'] = dc['label'][t:b, l:r]
        return dc

    def _calc_iou(self, box1: torch.Tensor, box2: torch.Tensor):
        inter = (
            torch.minimum(box1[:, None, 2:], box2[:, 2:]) - torch.maximum(box1[:, None, :2], box2[:, :2])
        ).clip(0).prod(dim=-1)
        area = (box2[:, 2:] - box2[:, :2]).prod(dim=-1)
        return inter / area


class RandomCrop(nn.Module):
    def __init__(self, size: list):
        super().__init__()
        self.crop_h, self.crop_w = size

    def forward(self, dc: dict) -> dict:
        _, h, w = dc['image'].shape
        wh = torch.Tensor([w, h])
        whcrop = torch.tensor([self.crop_w, self.crop_h], dtype=torch.int32)
        xymin = ((wh - whcrop) * torch.rand(2)).int()
        xymax = xymin + whcrop
        (l, t), (r, b) = xymin, xymax

        dc['image'] = dc['image'][:, t:b, l:r]
        if 'bboxes' in dc:
            dc['bboxes'][:, [0, 2]] -= l
            dc['bboxes'][:, [1, 3]] -= t
            dc['bboxes'][:, [0, 2]] = dc['bboxes'][:, [0, 2]].clip(min=0, max=r-l)
            dc['bboxes'][:, [1, 3]] = dc['bboxes'][:, [1, 3]].clip(min=0, max=b-t)
        if 'masks' in dc:
            dc['masks'] = dc['masks'][:, t:b, l:r]
        if 'label' in dc and dc['label'].ndim == 2:
            dc['label'] = dc['label'][t:b, l:r]
        return dc


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, dc: dict) -> dict:
        if random.random() < self.p:
            _, _, w = dc['image'].shape
            dc['image'] = dc['image'].flip(2)
            if 'bboxes' in dc:
                dc['bboxes'][:, [0, 2]] = w - dc['bboxes'][:, [2, 0]]
            if 'masks' in dc:
                dc['masks'] = dc['masks'].flip(2)
            if 'label' in dc and dc['label'].ndim == 2:
                dc['label'] = dc['label'].flip(1)
        return dc


class Resize(nn.Module):
    def __init__(self, size: list = None, scale_range: list = None):
        super().__init__()
        assert size or scale_range
        self.size = size
        self.scale_range = scale_range

    def forward(self, dc: dict) -> dict:
        _, h, w = dc['image'].shape
        if self.scale_range is not None:
            smin, smax = self.scale_range
            scale = (smax - smin) * random.random() + smin
            size = [int(h * scale), int(w * scale)]
        else:
            size = self.size
        dc['image'] = T.functional.resize(
            dc['image'],
            size=size,
            interpolation=T.InterpolationMode.BILINEAR
        )
        if 'bboxes' in dc:
            new_h, new_w = size
            dc['bboxes'][:, [0, 2]] *= (new_w / w)
            dc['bboxes'][:, [1, 3]] *= (new_h / h)
        if 'masks' in dc:
            dc['masks'] = T.functional.resize(
                dc['masks'],
                size=size,
                interpolation=T.InterpolationMode.NEAREST
            )
        if 'label' in dc and dc['label'].ndim == 2:
            dc['label'] = T.functional.resize(
                dc['label'].unsqueeze(0),
                size=size,
                interpolation=T.InterpolationMode.NEAREST
            ).squeeze()
        return dc
