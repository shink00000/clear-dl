from .extra_pool import ExtraPool
from .extra_conv import ExtraConv

EXTRAS = {
    'ExtraPool': ExtraPool,
    'ExtraConv': ExtraConv
}


def build_extra(extra: dict):
    return EXTRAS[extra.pop('type')](**extra)
