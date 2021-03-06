import torch.nn as nn


class TopDownGate(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.resize = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, x_above=None):
        if x_above is not None:
            x_above = self.resize(x_above)
            x = x + x_above
        return self.out(x)


class FPN(nn.Module):
    """ Feature Pyramid Network

    Forward Example:
        INPUT -- feat_7 --> conv  --> out_7 --
              |               |              |
              -- feat_6 --> ua+cv --> out_6 --
              |               |              |
              -- feat_5 --> ua+cv --> out_5 --
              |               |              |
              -- feat_4 --> ua+cv --> out_4 --
              |               |              |
              -- feat_3 --> ua+cv --> out_3 ----> OUTPUT

    Args:
        feats (Dict[int, torch.Tensor]): {3: feat_3, 4: feat_4, 5: feat_5, ...}
                                         they have same channel

    Returns:
        Dict[int, torch.Tensor]: {3: out_3, 4: out_4, 5: out_5, ...}
    """

    def __init__(self, feat_levels: list, in_channels: int, out_channels: int):
        super().__init__()
        self.feat_levels = feat_levels
        for level in feat_levels:
            setattr(self, f'f{level}', TopDownGate(in_channels, out_channels))

        self._init_weights()

    def forward(self, feats: dict) -> dict:
        out_feats = {}
        cur_feat = None
        for level in sorted(self.feat_levels, reverse=True):
            cur_feat = getattr(self, f'f{level}')(x=feats[level], x_above=cur_feat)
            out_feats[level] = cur_feat
        return out_feats

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
