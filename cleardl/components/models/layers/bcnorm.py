import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, mode: str = 'micro'):
        assert mode in ('large', 'micro')

        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.mode = mode

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        x = x.view(B, C, -1)
        if self.training:
            with torch.no_grad():
                m = x.mean(dim=(0, 2))
                v = x.var(dim=(0, 2), unbiased=False)
                self.running_mean += self.momentum * (m - self.running_mean)
                self.running_var += self.momentum * (v - self.running_var)

        w = self.weight.view(1, -1, 1)
        b = self.bias.view(1, -1, 1)
        if self.mode == 'large' and self.training:
            m = x.mean(dim=(0, 2), keepdim=True)
            v = x.var(dim=(0, 2), unbiased=False, keepdim=True)
        else:
            m = self.running_mean.view(1, -1, 1)
            v = self.running_var.view(1, -1, 1)
        y = w * torch.div(x - m, (v + self.eps).sqrt()) + b
        y = y.view(B, C, H, W)

        return y

    def __repr__(self):
        c_name = self.__class__.__name__
        args = []
        for arg in ['num_features', 'eps', 'momentum', 'mode']:
            args.append(f'{arg}={getattr(self, arg)}')
        args = ', '.join(args)
        return f'{c_name}({args})'


class ChannelNorm(nn.Module):
    def __init__(self, num_groups: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(num_groups))
        self.bias = nn.Parameter(torch.zeros(num_groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        G = self.num_groups

        x = x.view(B, G, -1)
        w = self.weight.view(1, -1, 1)
        b = self.bias.view(1, -1, 1)
        m = x.mean(dim=2, keepdim=True)
        v = x.var(dim=2, unbiased=False, keepdim=True)
        y = w * torch.div(x - m, (v + self.eps).sqrt()) + b
        y = y.view(B, C, H, W)

        return y

    def __repr__(self):
        c_name = self.__class__.__name__
        args = []
        for arg in ['num_groups', 'eps']:
            args.append(f'{arg}={getattr(self, arg)}')
        args = ', '.join(args)
        return f'{c_name}({args})'


class BCNorm(nn.Module):
    def __init__(self, num_features, num_groups, eps=1e-5, momentum=0.1, mode: str = 'micro'):
        super().__init__()
        self.bn = BatchNorm(num_features, eps=eps, momentum=momentum, mode=mode)
        self.cn = ChannelNorm(num_groups, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        out = self.cn(x)
        return out
