import torch
from torchmetrics import Metric


class MeanIoU(Metric):
    """
    Inputs:
        preds  : (N, C, H, W) pixelwise probability map
        targets: (N, H, W) pixelwise label

    Outputs:
        mean iou(s)
    """

    def __init__(self, n_classes: int, classwise: bool = False, class_names: list = None):
        super().__init__()

        self.n_classes = n_classes  # include background
        self.classwise = classwise
        self.class_names = class_names
        self.add_state('inters', default=torch.zeros(n_classes-1), dist_reduce_fx='sum')
        self.add_state('unions', default=torch.zeros(n_classes-1), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.argmax(dim=1).flatten()
        targets = targets.flatten()

        valid = targets > 0
        preds, targets = preds[valid], targets[valid]

        class_ids = torch.arange(self.n_classes, device=self.device)
        preds = preds.unsqueeze(1).repeat(1, self.n_classes) == class_ids
        targets = targets.unsqueeze(1).repeat(1, self.n_classes) == class_ids

        inters = (preds * targets).sum(dim=0)
        unions = preds.sum(dim=0) + targets.sum(dim=0) - inters

        self.inters += inters[1:]
        self.unions += unions[1:]

    def compute(self):
        if self.classwise:
            ious = torch.div(self.inters, self.unions).cpu().numpy()
            return {f'@{n}': v for n, v in zip(self.class_names, ious)}
        else:
            return torch.div(self.inters, self.unions).mean().item()
