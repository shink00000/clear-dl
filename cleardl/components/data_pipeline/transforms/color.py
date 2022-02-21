import torch
import torch.nn as nn
import random
import torchvision.transforms as T


class PhotoMetricDistortion(nn.Module):
    def __init__(self, brightness: float = 0, contrast: float = 0, saturation: float = 0, hue: float = 0):
        super().__init__()
        self.cj = T.ColorJitter(brightness, contrast, saturation, hue)

    def forward(self, dc: dict) -> dict:
        image = dc['image']
        image = self.cj(image)
        if random.randint(0, 1):
            image = image[torch.randperm(3)]
        dc['image'] = image
        return dc


class Normalize(nn.Module):
    def __init__(self, mean: list, std: list):
        super().__init__()
        self.normalize = T.Normalize(mean, std)

    def forward(self, dc: dict) -> dict:
        image = dc['image']
        dc['image'] = self.normalize(image)
        return dc
