import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..losses import build_loss


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.add_module('0', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.add_module('1', nn.BatchNorm2d(out_channels))
        self.add_module('2', nn.ReLU(inplace=True))


class Stage(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, n_layers: int = 2):
        super().__init__()
        self.add_module('0', ConvBNAct(in_channels, out_channels))
        for i in range(1, n_layers):
            self.add_module(str(i), ConvBNAct(out_channels, out_channels))


class UNet(nn.Module):
    def __init__(self, stage_channels: list, n_classes: int, criterion: dict):
        super().__init__()

        # layers
        assert len(stage_channels) == 5
        c1, c2, c3, c4, c5 = stage_channels
        self.stage1_down = Stage(3, c1)
        self.stage2_down = Stage(c1, c2)
        self.stage3_down = Stage(c2, c3)
        self.stage4_down = Stage(c3, c4)
        self.stage5 = Stage(c4, c5)
        self.stage4_up = Stage(c5, c4)
        self.stage3_up = Stage(c4, c3)
        self.stage2_up = Stage(c3, c2)
        self.stage1_up = Stage(c2, c1)

        self.pool = nn.MaxPool2d(2, 2)
        self.upconv_5to4 = nn.ConvTranspose2d(c5, c4, 2, 2)
        self.upconv_4to3 = nn.ConvTranspose2d(c4, c3, 2, 2)
        self.upconv_3to2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.upconv_2to1 = nn.ConvTranspose2d(c2, c1, 2, 2)

        self.cls_top = nn.Conv2d(c1, n_classes, kernel_size=1)

        # loss
        self.cls_loss = build_loss(criterion['cls_loss'])

        self._init_weights()

    def forward(self, x):
        x = x1 = self.stage1_down(x)
        x = self.pool(x)
        x = x2 = self.stage2_down(x)
        x = self.pool(x)
        x = x3 = self.stage3_down(x)
        x = self.pool(x)
        x = x4 = self.stage4_down(x)
        x = self.pool(x)
        x = self.upconv_5to4(self.stage5(x))
        x = self.upconv_4to3(self.stage4_up(torch.cat([x4, x], dim=1)))
        x = self.upconv_3to2(self.stage3_up(torch.cat([x3, x], dim=1)))
        x = self.upconv_2to1(self.stage2_up(torch.cat([x2, x], dim=1)))
        out = self.cls_top(self.stage1_up(torch.cat([x1, x], dim=1)))
        return out

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    if 'cls_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.cls_loss(outputs, targets)
        return loss

    def predict(self, outputs: torch.Tensor) -> torch.Tensor:
        preds = F.softmax(outputs, dim=1)
        return preds
