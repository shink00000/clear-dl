from torch.optim.lr_scheduler import LambdaLR


class PolynomialScheduler(LambdaLR):
    def __init__(self, optimizer, gamma: float, n_iterations: int):
        def lr_lambda(iter): return (1 - min(iter, n_iterations) / n_iterations) ** gamma
        super().__init__(optimizer, lr_lambda)
