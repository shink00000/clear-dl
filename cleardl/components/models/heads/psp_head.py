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


class PSPHead(nn.Module):
    def __init__(self, in_channels: int, aux_in_channels: int, mid_channels: int,
                 aux_mid_channels: int, n_classes: int):
        super().__init__()
        self.cls_branch = Branch(in_channels, mid_channels)
        self.aux_branch = Branch(aux_in_channels, aux_mid_channels)
        self.cls_top = nn.Conv2d(mid_channels, n_classes, kernel_size=1)
        self.aux_top = nn.Conv2d(aux_mid_channels, n_classes, kernel_size=1)

        self._init_weights()

    def forward(self, x: torch.Tensor, aux: torch.Tensor):
        cls_outs = self.cls_top(self.cls_branch(x))
        aux_outs = self.aux_top(self.aux_branch(aux))
        return cls_outs, aux_outs

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
