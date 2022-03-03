import torch.nn as nn

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


def replace_bn_to_gn(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            tm = model
            for attr in name.split('.'):
                target_module = tm
                if attr.isdigit():
                    tm = tm[int(attr)]
                else:
                    tm = getattr(tm, attr)
            gn = nn.GroupNorm(GROUP_NORM_LOOKUP[tm.num_features], tm.num_features)
            nn.init.constant_(gn.weight, 1.0)
            nn.init.constant_(gn.bias, 0.0)
            setattr(target_module, attr, gn)
