from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iterations, last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.max_iterations = max_iterations
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return [base_lr for base_lr in self.base_lrs]
        return [
            base_lr * (1 - min(self.last_epoch, self.max_iterations - 1) / self.max_iterations) ** self.gamma
            for base_lr in self.base_lrs
        ]

    def _get_closed_form_lr(self):
        return [
            base_lr * (1 - min(self.last_epoch, self.max_iterations - 1) / self.max_iterations) ** self.gamma
            for base_lr in self.base_lrs
        ]
