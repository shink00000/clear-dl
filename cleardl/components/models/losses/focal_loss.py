import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """ Focal Loss

    Args:
        input (torch.Tensor): [N, C]
        target (torch.Tensor): [N, C]

    Examples:
        >>> focal_loss = FocalLoss(reduction='mean')
        >>> input = torch.rand(N, C)
        >>> target = torch.rand(N, C)
        >>> loss = focal_loss(input, target)

    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        target = F.one_hot(target, num_classes=input.size(-1)+1)[..., 1:].type_as(input)
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
