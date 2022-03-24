import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """ Sigmoid Focal Loss

    Args:
        input (torch.Tensor): [N, C]
        target (torch.Tensor): [N, C]

    Examples:
        >>> criterion = FocalLoss(reduction='mean')
        >>> input = torch.rand(N, C)
        >>> target = torch.rand(N, C)
        >>> loss = criterion(input, target)

    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = 'mean', skip_first: bool = True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.skip_first = skip_first

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.skip_first:
            target = F.one_hot(target, num_classes=input.size(-1)+1)[..., 1:].type_as(input)
        else:
            target = F.one_hot(target, num_classes=input.size(-1)).type_as(input)
        pt = (1 - input.sigmoid()) * target + input.sigmoid() * (1 - target)
        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        loss = at * pt.pow(self.gamma) * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SoftmaxFocalLoss(nn.Module):
    """ Softmax Focal Loss

    Args:
        input (torch.Tensor): [N, C, H, W]
        target (torch.Tensor): [N, H, W]

    Examples:
        >>> criterion = SoftmaxFocalLoss(reduction='mean')
        >>> input = torch.rand(N, C, H, W)
        >>> target = torch.randint(0, C, (N, H, W))
        >>> loss = criterion(input, target)

    """

    def __init__(self, gamma: float = 2, ignore_index: int = -1, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t = target.detach().clone()
        valid = t != self.ignore_index
        t[~valid] = 0
        p = F.softmax(input, dim=1).gather(dim=1, index=t.unsqueeze(1)).squeeze()
        loss = F.cross_entropy(input, target, ignore_index=self.ignore_index, reduction='none')
        loss = (1 - p).pow(self.gamma) * loss

        if self.reduction == 'mean':
            return loss[valid].mean()
        elif self.reduction == 'sum':
            return loss[valid].sum()
        else:
            return loss
