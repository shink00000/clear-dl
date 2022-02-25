import torch
import torch.nn as nn
import numpy as np


class Branch(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )


class AuxiliaryHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        channels = 256
        self.aux_branch = Branch(in_channels, channels)
        self.aux_top = nn.Conv2d(channels, n_classes, kernel_size=1)

        self._init_weights()

    def forward(self, x: torch.Tensor):
        x = self.aux_branch(x)
        out = self.aux_top(x)
        return out

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    if '_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
