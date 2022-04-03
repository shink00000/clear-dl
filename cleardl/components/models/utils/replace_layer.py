import torch.nn as nn

from ..layers.conv_ws import Conv2dWS, ConvTranspose2dWS
from ..layers.bcnorm import BCNorm


def replace_layer_(model, mode, **kwargs):
    if mode == 'WS+GN':
        replace_bn_to_wsgn_(model, **kwargs)
    elif mode == 'WS+BCN':
        replace_bn_to_wsbcn_(model, **kwargs)


def replace_bn_to_wsgn_(model, num_groups: int = 32):
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
            gn = nn.GroupNorm(min(num_groups, m.num_features//4), m.num_features)
            gn.weight = m.weight
            gn.bias = m.bias
            setattr(module, attr, gn)


def replace_bn_to_wsbcn_(model, num_groups: int = 32):
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
            bcn = BCNorm(m.num_features, min(num_groups, m.num_features//4))
            bcn.bn.weight = m.weight
            bcn.bn.bias = m.bias
            nn.init.constant_(bcn.cn.weight, 1.0)
            nn.init.constant_(bcn.cn.bias, 0.0)
            setattr(module, attr, bcn)
