import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101
)


class ResNet(nn.Module):
    def __init__(self, depth: int, n_classes: int = 1000, weights: str = 'default',
                 frozen_stages: int = -1, **kwargs):
        super().__init__()

        if weights == 'default':
            base = self._build_base(depth)(pretrained=True, **kwargs)
        else:
            base = self._build_base(depth)(pretrained=False, **kwargs)
            if weights is not None:
                base.load_state_dict(torch.load(weights, map_location='cpu'))

        freeze = frozen_stages >= 0
        for key, m in base.named_children():
            setattr(self, key, m)
            if freeze:
                if 'layer' in key and int(key[-1]) > frozen_stages:
                    freeze = False
                    continue
                for p in m.parameters():
                    p.requires_grad = False

        if self.fc.out_features != n_classes:
            self.fc = nn.Linear(self.fc.in_features, n_classes, bias=True)
            nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.layer1(self.maxpool(x1))
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.avgpool(x5)
        x = x.flatten(1)
        return self.fc(x)

    def _build_base(self, depth: int):
        return {
            18: resnet18,
            34: resnet34,
            50: resnet50,
            101: resnet101,
        }[depth]
