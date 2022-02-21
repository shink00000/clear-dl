from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiStepLR, ExponentialLR


class WarmupMultiStepLR(SequentialLR):
    def __init__(self, optimizer, milestones: list, gamma: float = 0.1,
                 warmup_interval: int = 3, warmup_start_factor: float = 0.3):
        milestones = [milestone-warmup_interval for milestone in milestones]
        warmup_scheduler = LinearLR(optimizer, warmup_start_factor, 1.0, total_iters=warmup_interval)
        ordinary_scheduler = MultiStepLR(optimizer, milestones, gamma)
        super().__init__(optimizer, [warmup_scheduler, ordinary_scheduler], milestones=[warmup_interval])

    def get_last_lr(self):
        if self.last_epoch < self._milestones[0]:
            return self._schedulers[0].get_last_lr()
        else:
            return self._schedulers[1].get_last_lr()


class WarmupExponentialLR(SequentialLR):
    def __init__(self, optimizer, gamma: float, warmup_interval: int = 3, warmup_start_factor: float = 0.3):
        warmup_scheduler = LinearLR(optimizer, warmup_start_factor, 1.0, total_iters=warmup_interval)
        ordinary_scheduler = ExponentialLR(optimizer, gamma)
        super().__init__(optimizer, [warmup_scheduler, ordinary_scheduler], milestones=[warmup_interval])

    def get_last_lr(self):
        if self.last_epoch < self._milestones[0]:
            return self._schedulers[0].get_last_lr()
        else:
            return self._schedulers[1].get_last_lr()
