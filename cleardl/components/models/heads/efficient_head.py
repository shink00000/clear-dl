import torch
import torch.nn as nn
import numpy as np

from ..layers.separatable_conv import SeparatableConv2d


class Branch(nn.Sequential):
    def __init__(self, channels, n_stacks):
        super().__init__()
        for i in range(n_stacks):
            block = nn.Sequential(
                SeparatableConv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True)
            )
            self.add_module(str(i), block)


def update_branch(branch: nn.Module, base_branch: nn.Module):
    for i in range(len(branch)):
        branch[i][0] = base_branch[i][0]


class EfficientHead(nn.Module):
    def __init__(self, feat_sizes: list, in_channels: int, n_classes: int, n_stacks: int = 4):
        super().__init__()
        self.feat_sizes = feat_sizes
        self.n_classes = n_classes

        for type in ['reg', 'cls']:
            base_branch = Branch(in_channels, n_stacks)
            for fsize in feat_sizes:
                branch = Branch(in_channels, n_stacks)
                update_branch(branch, base_branch)
                setattr(self, f'{type}_branch_feat_{fsize}', branch)
        self.reg_top = nn.Conv2d(in_channels, 9*4, kernel_size=3, padding=1)
        self.cls_top = nn.Conv2d(in_channels, 9*n_classes, kernel_size=3, padding=1)

        self._init_weights()

    def forward(self, feats: dict):
        reg_outs, cls_outs = [], []
        for fsize in self.feat_sizes:
            reg_feat = getattr(self, f'reg_branch_feat_{fsize}')(feats[fsize])
            cls_feat = getattr(self, f'cls_branch_feat_{fsize}')(feats[fsize])
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
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


class UpConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )


class EfficientSegHead(nn.Module):
    def __init__(self, feat_sizes: list, in_channels: int, n_classes: int, output_size: list):
        super().__init__()
        self.feat_sizes = feat_sizes

        for fsize in feat_sizes:
            setattr(self, f'uc{fsize}', UpConvBlock(in_channels, in_channels))
        self.cls_top = nn.Conv2d(in_channels, n_classes, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(output_size)

        self._init_weights()

    def forward(self, feats: dict):
        x = None
        for fsize in sorted(self.feat_sizes, reverse=True):
            if x is None:
                x = feats[fsize]
            else:
                x = feats[fsize] + x
            x = getattr(self, f'uc{fsize}')(x)
        x = self.cls_top(x)
        out = self.up(x)
        return out

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    if 'cls_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
