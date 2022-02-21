import torch
import torch.nn as nn


class PixelwiseCutOff(nn.Module):
    def __init__(self, p: list = [0, 0.05]):
        super().__init__()
        if isinstance(p, list):
            self.pmin, self.pmax = p
        else:
            self.pmin = self.pmax = p

    def forward(self, dc: dict) -> dict:
        _, h, w = dc['image'].shape
        survive = torch.rand(1, h, w) > torch.empty(1).uniform_(self.pmin, self.pmax)
        dc['image'] *= survive
        return dc
