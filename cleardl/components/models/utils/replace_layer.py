import torch.nn as nn

from ..layers.conv_ws import Conv2dWS, ConvTranspose2dWS

GROUP_NORM_LOOKUP = {
    16: 2,  # -> channels per group: 8
    32: 4,  # -> channels per group: 8
    64: 8,  # -> channels per group: 8
    128: 8,  # -> channels per group: 16
    256: 16,  # -> channels per group: 16
    512: 32,  # -> channels per group: 16
    1024: 32,  # -> channels per group: 32
    2048: 32,  # -> channels per group: 64
}


def replace_bn_to_wsgn_(model):
    def _module_attr(name):
        m_ = model
        for attr in name.split('.'):
            module = m_
            if attr.isdigit():
                m_ = m_[int(attr)]
            else:
                m_ = getattr(m_, attr)
        return module, attr

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.bias is None:
            module, attr = _module_attr(name)
            conv_ws = Conv2dWS(
                m.in_channels,
                m.out_channels,
                m.kernel_size,
                stride=m.stride,
                padding=m.padding,
                dilation=m.dilation,
                groups=m.groups,
                bias=False)
            conv_ws.weight = m.weight
            setattr(module, attr, conv_ws)

        elif isinstance(m, nn.ConvTranspose2d) and m.bias is None:
            module, attr = _module_attr(name)
            conv_ws = ConvTranspose2dWS(
                m.in_channels,
                m.out_channels,
                m.kernel_size,
                stride=m.stride,
                padding=m.padding,
                output_padding=m.output_padding,
                dilation=m.dilation,
                groups=m.groups,
                bias=False)
            conv_ws.weight = m.weight
            setattr(module, attr, conv_ws)

        elif isinstance(m, nn.BatchNorm2d):
            module, attr = _module_attr(name)
            gn = nn.GroupNorm(GROUP_NORM_LOOKUP.get(m.num_features, 8), m.num_features)
            gn.weight = m.weight
            gn.bias = m.bias
            setattr(module, attr, gn)
