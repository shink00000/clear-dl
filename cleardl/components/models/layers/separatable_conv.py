import torch.nn as nn


class SeparatableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
