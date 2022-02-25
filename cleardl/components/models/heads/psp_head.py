import torch
import torch.nn as nn
import numpy as np


class PSPHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.drop = nn.Dropout2d(0.1)
        self.cls_top = nn.Conv2d(in_channels, n_classes, kernel_size=1)

        self._init_weights()

    def forward(self, x: torch.Tensor):
        x = self.drop(x)
        cls_outs = self.cls_top(x)
        return cls_outs

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    if '_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)
