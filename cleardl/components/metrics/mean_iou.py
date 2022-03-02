import torch
import torch.nn.functional as F
import json
from torchmetrics import Metric


class MeanIoU(Metric):
    """
    Inputs:
        preds  : (N, C, H, W) pixelwise probability map
        metas: (N, H, W) pixelwise label

    Outputs:
        mean iou(s)
    """

    def __init__(self, n_classes: int, labelmap_path: str, ignore_index: int = -1):
        super().__init__()

        self.n_classes = n_classes
        with open(labelmap_path, 'r') as f:
            self.labelmap = json.load(f, object_hook=lambda d: {int(k): v for k, v in d.items()})
        self.length = max(len(v) for v in self.labelmap.values())
        self.ignore_index = ignore_index

        self.add_state('cm', default=torch.zeros(n_classes, n_classes), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, metas: torch.Tensor):
        preds = F.interpolate(preds.cpu(), size=metas.shape[1:], mode='bilinear', align_corners=True)
        preds = preds.argmax(dim=1).flatten()
        metas = metas.flatten()

        valid = metas != self.ignore_index
        preds, metas = preds[valid], metas[valid]

        counts = (metas * self.n_classes + preds).bincount(minlength=self.n_classes**2)
        cm = counts.reshape(self.n_classes, self.n_classes)
        self.cm += cm

    def compute(self) -> dict:
        tp = self.cm.diag()
        fp = self.cm.sum(dim=1) - tp
        fn = self.cm.sum(dim=0) - tp

        ious = torch.div(tp, tp + fp + fn).numpy()

        lines = []
        for i in range(self.n_classes):
            line = f'{self.labelmap[i]:{self.length}}: {ious[i]:.04f}\n'
            lines.append(line)
        name = 'mean'
        line = f'\n{name:{self.length}}: {ious.mean():.04f}\n'
        lines.append(line)

        return {
            'text': ''.join(lines),
            'mIoU': ious.mean()
        }
