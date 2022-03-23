from typing import Generator
from torchmetrics import MetricCollection

from .components.data_pipeline import DataPipeline
from .components.frameworks import FRAMEWORKS
from .components.models import MODELS
from .components.optimizers import OPTIMIZERS
from .components.schedulers import SCHEDULERS
from .components.metrics import METRICS


def build_data_pipeline(cfg: dict):
    return DataPipeline(**cfg)


def build_framework(cfg: dict):
    Frame = FRAMEWORKS[cfg['type']]
    Frame.INPUT_SIZE = cfg['input_size']
    return Frame


def build_model(cfg: dict):
    return MODELS[cfg.pop('type')](**cfg)


def build_optimizer(cfg: dict, named_parameters: Generator[tuple, None, None]):
    param_args = cfg.pop('param_args', {})
    type = cfg.pop('type')
    param_groups = []
    for name, p in named_parameters:
        if p.requires_grad:
            param = {'params': [p], **cfg.copy()}
            for key, multi_list in param_args.items():
                for substr, multi in multi_list.items():
                    if substr in name:
                        param[key] *= multi
            if 'weight_decay' in param and p.ndim == 1:
                param['weight_decay'] = 0
            param_groups.append(param)
    return OPTIMIZERS[type](param_groups, **cfg)


def build_scheduler(cfg: dict, optimizer):
    return SCHEDULERS[cfg.pop('type')](optimizer, **cfg)


def build_metrics(cfg: list):
    return MetricCollection(
        [METRICS[c.pop('type')](**c) for c in cfg]
    )
