from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR

from .warmup_scheduler import (
    WarmupMultiStepLR,
    WarmupExponentialLR
)
from .polynomial_scheduler import PolynomialScheduler


SCHEDULERS = {
    'MultiStepLR': MultiStepLR,
    'ExponentialLR': ExponentialLR,
    'WarmupMultiStepLR': WarmupMultiStepLR,
    'WarmupExponentialLR': WarmupExponentialLR,
    'PolynomialScheduler': PolynomialScheduler
}
