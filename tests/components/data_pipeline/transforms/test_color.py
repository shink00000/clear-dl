import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from pathlib import Path

from cleardl.components.data_pipeline.transforms.color import PhotoMetricDistortion


def test_photometric_distortion():
    image = np.zeros((256, 256, 3), dtype='uint8')
    image = cv2.circle(image, (64, 64), 32, (255, 0, 0), -1)
    image = cv2.circle(image, (64*3, 64*3), 32, (0, 0, 255), -1)
    mask = np.zeros((256, 256, 3), dtype='uint8')
    mask = cv2.circle(mask, (64, 64), 32, (255, 0, 0), 1)
    mask = cv2.circle(mask, (64*3, 64*3), 32, (0, 0, 255), 1)

    dc = {
        'image': torch.from_numpy(image).float().permute(2, 0, 1) / 255,
        'bboxes': torch.Tensor([[32, 32, 96, 96], [160, 160, 224, 224]]),
        'labels': torch.LongTensor([1, 2]),
        'masks': torch.from_numpy(np.stack([mask[..., 0] > 0, mask[..., 2] > 0])).float()
    }

    m = PhotoMetricDistortion(0.5, 0.5, 0.5, 0.5)
    dc = m(dc)

    image = (dc['image'].permute(1, 2, 0).numpy() * 255).astype('uint8')
    bboxes = dc['bboxes'].int().numpy()
    masks = list(dc['masks'].bool().numpy())
    for mask in masks:
        image = np.where(mask[..., None], np.full_like(image, 255), image)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(tuple(bbox), outline='white', width=2)
    image.save(Path(__file__).parent / 'test_photometric_distortion.png')
