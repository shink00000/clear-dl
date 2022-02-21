import torch
import torch.nn as nn
import numpy as np


class Branch(nn.Sequential):
    def __init__(self, channels: int, n_stacks: int):
        super().__init__()
        for i in range(n_stacks):
            block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.add_module(str(i), block)


class RetinaHead(nn.Module):
    def __init__(self, feat_sizes: list, in_channels: int, n_classes: int, n_stacks: int = 4):
        super().__init__()
        self.feat_sizes = feat_sizes
        self.n_classes = n_classes
        self.reg_branch = Branch(in_channels, n_stacks)
        self.cls_branch = Branch(in_channels, n_stacks)
        self.reg_top = nn.Conv2d(in_channels, 9*4, kernel_size=3, padding=1)
        self.cls_top = nn.Conv2d(in_channels, 9*n_classes, kernel_size=3, padding=1)

        self._init_weights()

    def forward(self, feats: dict):
        reg_outs, cls_outs = [], []
        for fsize in self.feat_sizes:
            reg_feat = self.reg_branch(feats[fsize])
            cls_feat = self.cls_branch(feats[fsize])
            reg_out = self.reg_top(reg_feat)
            cls_out = self.cls_top(cls_feat)
            reg_outs.append(reg_out.permute(0, 2, 3, 1).reshape(reg_out.size(0), -1, 4))
            cls_outs.append(cls_out.permute(0, 2, 3, 1).reshape(cls_out.size(0), -1, self.n_classes))
        reg_outs = torch.cat(reg_outs, dim=1)
        cls_outs = torch.cat(cls_outs, dim=1)

        return reg_outs, cls_outs

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    if 'cls_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)
