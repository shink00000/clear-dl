import torch
import json
from torchmetrics import Metric


class MeanIoU(Metric):
    """
    Inputs:
        preds  : (N, C, H, W) pixelwise probability map
        targets: (N, H, W) pixelwise label

    Outputs:
        mean iou(s)
    """

    def __init__(self, n_classes: int, labelmap_path: str):
        super().__init__()

        self.n_classes = n_classes  # include background
        with open(labelmap_path, 'r') as f:
            self.labelmap = json.load(f, object_hook=lambda d: {int(k): v for k, v in d.items()})
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

    def compute(self) -> dict:
        lines = []
        ious = torch.div(self.inters, self.unions).cpu().numpy()
        max_len = max(len(v) for v in self.labelmap.values())
        for i in range(1, self.n_classes):
            lines.append(f'{self.labelmap[i]:{max_len}}: {ious[i-1]:.04f}\n')
        text = ''.join(lines)

        return {
            'text': text,
            'mIoU': torch.div(self.inters, self.unions).mean().item()
        }
