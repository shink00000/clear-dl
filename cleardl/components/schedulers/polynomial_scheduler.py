from torch.optim.lr_scheduler import LambdaLR


class PolynomialScheduler(LambdaLR):
    def __init__(self, optimizer, min_factor: float, gamma: float, n_iterations: int):
        a = 1 - min_factor
        b = min_factor
        def lr_lambda(epoch): return a * (1 - min(epoch, n_iterations) / n_iterations) ** gamma + b
        super().__init__(optimizer, lr_lambda)
