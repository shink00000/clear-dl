import torch
import torch.nn as nn
import numpy as np


class Scale(nn.Module):
    def __init__(self, scale: float = 1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale

    def __repr__(self):
        return f'{self.__class__.__name__}(scale={self.scale})'


class Branch(nn.Sequential):
    def __init__(self, channels: int, n_stacks: int):
        super().__init__()
        for i in range(n_stacks):
            block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, channels),
                nn.ReLU(inplace=True)
            )
            self.add_module(str(i), block)


class FCOSHead(nn.Module):
    def __init__(self, feat_levels: list, in_channels: int, n_classes: int, n_stacks: int = 4):
        super().__init__()
        self.feat_levels = feat_levels
        self.n_classes = n_classes
        self.reg_branch = Branch(in_channels, n_stacks)
        self.cls_branch = Branch(in_channels, n_stacks)
        self.reg_top = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.cls_top = nn.Conv2d(in_channels, n_classes, kernel_size=3, padding=1)
        self.cnt_top = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self.scales = nn.ModuleDict({f'feat_{level}': Scale(1.0) for level in feat_levels})

        self._init_weights()

    def forward(self, feats: dict):
        reg_outs, cls_outs, cnt_outs = [], [], []
        for level in self.feat_levels:
            reg_feat = self.reg_branch(feats[level])
            cls_feat = self.cls_branch(feats[level])
            reg_out = self.scales[f'feat_{level}'](self.reg_top(reg_feat)).exp()
            cls_out = self.cls_top(cls_feat)
            cnt_out = self.cnt_top(cls_feat)
            reg_outs.append(reg_out.permute(0, 2, 3, 1).flatten(1, 2))
            cls_outs.append(cls_out.permute(0, 2, 3, 1).flatten(1, 2))
            cnt_outs.append(cnt_out.permute(0, 2, 3, 1).flatten(1, 2))
        reg_outs = torch.cat(reg_outs, dim=1)
        cls_outs = torch.cat(cls_outs, dim=1)
        cnt_outs = torch.cat(cnt_outs, dim=1)

        return reg_outs, cls_outs, cnt_outs

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    if 'cls_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
