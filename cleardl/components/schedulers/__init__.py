from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR

from .warmup_scheduler import (
    WarmupMultiStepLR,
    WarmupExponentialLR
)
from .polynomial_scheduler import PolynomialLR


SCHEDULERS = {
    'MultiStepLR': MultiStepLR,
    'ExponentialLR': ExponentialLR,
    'WarmupMultiStepLR': WarmupMultiStepLR,
    'WarmupExponentialLR': WarmupExponentialLR,
    'PolynomialLR': PolynomialLR
}
