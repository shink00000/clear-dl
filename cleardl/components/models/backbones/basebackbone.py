import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod


class BaseBackBone(nn.Module, metaclass=ABCMeta):
    """ BackBone

    Forward Example:
        1. extract intermediate tensors (= features) from XXXXX
            - argment 'feat_sizes' means 'feature size' to select
                - if feature size = n, the HW size of feature tensor is (input_HW / 2 ** n)
        2. use 1. to generate backbone features
            - if align_channel is True, backbone features must be same channel
            - when creating a smaller feature (that is, a larger feat_size),
                generate extra features according to extra_mode

    Returns:
        Dict[int, torch.Tensor]: {3: out_3, 4: out_4, 5: out_5} (example)
    """

    def __init__(self, feat_sizes: list, out_channels: int = 64, extra_mode: str = 'conv',
                 max_size: int = 5, align_channel: bool = True, **kwargs):
        super().__init__()
        assert extra_mode in ('conv', 'pool')

        self.feat_sizes = feat_sizes
        self.extra_mode = extra_mode
        self.max_size = max_size
        self.align_channel = align_channel

        self._build(**kwargs)
        if align_channel:
            self._build_aligner(out_channels)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> dict:
        feats = self._forward(x)
        if self.align_channel:
            feats = self._align(feats)
        else:
            feats = {fsize: feats[fsize] for fsize in self.feat_sizes}
        return feats

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> dict:
        pass

    def _align(self, feats: dict) -> dict:
        out_feats = {}
        for fsize in self.feat_sizes:
            if fsize in feats:
                feat = feats[fsize]
            elif self.extra_mode == 'conv':
                feat = feats[self.max_size] if fsize == self.max_size+1 else out_feats[fsize-1]
            elif self.extra_mode == 'pool':
                feat = out_feats[fsize-1]
            out_feat = self.aligner[f'feat_{fsize}'](feat)
            out_feats[fsize] = out_feat
        return out_feats

    @abstractmethod
    def _build(self, **kwargs) -> None:
        pass

    def get_channels(self) -> dict:
        with torch.no_grad():
            X = 2 ** self.max_size
            feats = self._forward(torch.rand(2, 3, X, X))
            feat_channels = {fsize: feat.size(1) for fsize, feat in feats.items()}
        return feat_channels

    def _build_aligner(self, out_channels: int):
        feat_channels = self.get_channels()
        self.aligner = nn.ModuleDict()
        for fsize in self.feat_sizes:
            if fsize <= self.max_size:
                align_module = nn.Conv2d(feat_channels[fsize], out_channels, kernel_size=1)
            else:
                if self.extra_mode == 'conv':
                    if fsize == self.max_size+1:
                        align_module = nn.Conv2d(
                            feat_channels[self.max_size], out_channels, kernel_size=3, stride=2, padding=1
                        )
                    else:
                        align_module = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                        )
                elif self.extra_mode == 'pool':
                    align_module = nn.MaxPool2d(2, 2)
            self.aligner[f'feat_{fsize}'] = align_module

    def _init_weights(self):
        for name, m in self.named_modules():
            if name.startswith('aligner') and isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
