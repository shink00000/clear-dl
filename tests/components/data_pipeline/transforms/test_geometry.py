import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from pathlib import Path

from cleardl.components.data_pipeline.transforms.geometry import (
    RandomExpand, RandomMinIoUCrop, RandomHorizontalFlip, Resize, RandomMinAreaCrop
)


def base_function(module) -> Image:
    image = np.zeros((256, 256, 3), dtype='uint8')
    image = cv2.circle(image, (64, 64), 32, (255, 0, 0), -1)
    image = cv2.circle(image, (64*3, 64*3), 32, (0, 0, 255), -1)
    mask = np.zeros((256, 256, 3), dtype='uint8')
    mask = cv2.circle(mask, (64, 64), 32, (255, 0, 0), 1)
    mask = cv2.circle(mask, (64*3, 64*3), 32, (0, 0, 255), 1)
    label = np.zeros((256, 256), dtype='uint8')
    label = cv2.circle(label, (64, 64), 32, 100, -1)
    label = cv2.circle(label, (64*3, 64*3), 32, 200, -1)

    dc = {
        'image': torch.from_numpy(image).float().permute(2, 0, 1) / 255,
        'bboxes': torch.Tensor([[32, 32, 96, 96], [160, 160, 224, 224]]),
        'labels': torch.LongTensor([1, 2]),
        'masks': torch.from_numpy(np.stack([mask[..., 0] > 0, mask[..., 2] > 0])).float(),
        'label': torch.from_numpy(label).long()
    }

    dc = module(dc)

    image = (dc['image'].permute(1, 2, 0).numpy() * 255).astype('uint8')
    bboxes = dc['bboxes'].int().numpy()
    masks = list(dc['masks'].bool().numpy())
    label = dc['label'].numpy().astype('uint8')
    for mask in masks:
        image = np.where(mask[..., None], np.full_like(image, 255), image)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(tuple(bbox), outline='white', width=2)
    label = Image.fromarray(label)
    ret = Image.new('RGB', (image.width*2, image.height))
    ret.paste(image, (0, 0))
    ret.paste(label, (image.width, 0))
    return ret


def test_random_expand():
    for n in range(5):
        image = base_function(RandomExpand(p=1))
        image.save(Path(__file__).parent/f'test_random_expand_{n}.png')


def test_random_min_iou_crop():
    for n in range(10):
        image = base_function(RandomMinIoUCrop(min_ious=[0.0, 0.3, 0.5, 0.9], p=1))
        image.save(Path(__file__).parent/f'test_random_min_iou_crop_{n}.png')


def test_random_min_area_crop():
    for n in range(10):
        image = base_function(RandomMinAreaCrop(min_areas=[0.3, 0.5, 0.9], p=1))
        image.save(Path(__file__).parent/f'test_random_min_area_crop_{n}.png')


def test_random_horizontal_flip():
    image = base_function(RandomHorizontalFlip(p=1))
    image.save(Path(__file__).parent/'test_random_horizontal_flip.png')


def test_resize():
    image = base_function(Resize(size=[128, 128]))
    image.save(Path(__file__).parent/'test_resize.png')
