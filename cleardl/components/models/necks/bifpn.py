import torch
import torch.nn as nn

from ..layers.separatable_conv import SeparatableConv2d


class TopDownGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.out = nn.Sequential(
            SeparatableConv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.resize = nn.Upsample(scale_factor=2, mode='nearest')
        self.gate_weights = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, x_above):
        x_above = self.resize(x_above)
        weights = self.relu(self.gate_weights)
        x = torch.div(
            x * weights[0] + x_above * weights[1],
            weights.sum() + 1e-4
        )
        return self.out(x)


class BottomUpGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.out = nn.Sequential(
            SeparatableConv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.resize = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gate_weights = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, x_below, x_mid=None):
        x_below = self.resize(x_below)
        weights = self.relu(self.gate_weights)
        if x_mid is None:
            xs = x * weights[0] + x_below * weights[1]
        else:
            xs = x * weights[0] + x_below * weights[1] + x_mid + weights[2]
        x = torch.div(
            xs,
            weights.sum() + 1e-4
        )
        return self.out(x)


class BiFPNBlock(nn.Module):
    def __init__(self, feat_sizes: list, channels: int):
        super().__init__()
        self.feat_sizes = feat_sizes
        for fsize in feat_sizes[:-1]:
            setattr(self, f'top_down_feat_{fsize}', TopDownGate(channels))
        for fsize in feat_sizes[1:]:
            setattr(self, f'bottom_up_feat_{fsize}', BottomUpGate(channels))

    def forward(self, feats: dict):
        out_feats, mid_feats = {}, {}
        cur_feat = None
        for fsize in sorted(self.feat_sizes, reverse=True):
            if cur_feat is None:
                cur_feat = feats[fsize]
                # not register the top feat to mid_feats
            else:
                cur_feat = getattr(self, f'top_down_feat_{fsize}')(
                    x=feats[fsize],
                    x_above=cur_feat
                )
                mid_feats[fsize] = cur_feat
        cur_feat = None
        for fsize in sorted(self.feat_sizes, reverse=False):
            if cur_feat is None:
                cur_feat = mid_feats[fsize]
            else:
                cur_feat = getattr(self, f'bottom_up_feat_{fsize}')(
                    x=feats[fsize],
                    x_below=cur_feat,
                    x_mid=mid_feats.get(fsize, None),
                )
            out_feats[fsize] = cur_feat
        return out_feats


class BiFPN(nn.Sequential):
    """ Bi-directional Feature Pyramid Network

    Forward Example:
        INPUT -- feat_7 ----------------------> da+cv --> out_7 --
              |               |                   |              |
              -- feat_6 --> ua+cv --> mid_6 --> da+cv --> out_6 --
              |               |                   |              |
              -- feat_5 --> ua+cv --> mid_5 --> da+cv --> out_5 --
              |               |                   |              |
              -- feat_4 --> ua+cv --> mid_4 --> da+cv --> out_4 --
              |               |                   |              |
              -- feat_3 --> ua+cv --> mid_3 ------------> out_3 ----> (repeat) --> OUTPUT

    Args:
        feats (Dict[int, torch.Tensor]): {3: feat_3, 4: feat_4, 5: feat_5, ...}
                                         they have same channel

    Returns:
        Dict[int, torch.Tensor]: {3: out_3, 4: out_4, 5: out_5, ...}
    """

    def __init__(self, feat_sizes: list, channels: int, n_blocks: int = 3):
        super().__init__()
        for i in range(n_blocks):
            self.add_module(str(i), BiFPNBlock(feat_sizes, channels))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
