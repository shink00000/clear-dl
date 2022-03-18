from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iterations, last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.max_iterations = max_iterations
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [
            group['initial_lr'] * (1 - min(self.last_epoch, self.max_iterations - 1) /
                                   self.max_iterations) ** self.gamma
            for group in self.optimizer.param_groups
        ]
