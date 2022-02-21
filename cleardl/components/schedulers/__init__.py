from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR

from .warmup_scheduler import (
    WarmupMultiStepLR,
    WarmupExponentialLR
)


SCHEDULERS = {
    'MultiStepLR': MultiStepLR,
    'ExponentialLR': ExponentialLR,
    'WarmupMultiStepLR': WarmupMultiStepLR,
    'WarmupExponentialLR': WarmupExponentialLR
}
