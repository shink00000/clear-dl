import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod


class BaseFramework(nn.Module, metaclass=ABCMeta):
    def __init__(self, model: object, optimizer: object, scheduler: object, metrics: object):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics

        self._results = {'train': {}, 'val': {}}
        self._checkpoints = {}
        self._last_epoch = 0
        self._run_eval = False

        self._train_loss = 0
        self._train_counts = 0
        self._val_loss = 0
        self._val_counts = 0

    @property
    def run_eval(self):
        return self._run_eval

    @run_eval.setter
    def run_eval(self, run_eval):
        self._run_eval = run_eval

    def classwise_eval_(self, classwise_eval: bool):
        for k, m in self.metrics.items():
            if hasattr(m, 'classwise'):
                m.classwise = classwise_eval
                print(f'{k} will be a classwise metric')

    @property
    def results(self):
        return self._results

    @property
    def checkpoints(self):
        return self._checkpoints

    @property
    def last_epoch(self):
        return self._last_epoch

    @property
    def train_loss(self):
        return self._train_loss

    @property
    def val_loss(self):
        return self._val_loss

    @abstractmethod
    def epoch_start(self):
        pass

    @abstractmethod
    def train_step(self, data: tuple):
        pass

    @abstractmethod
    def train_step_end(self):
        pass

    @abstractmethod
    def val_step(self, data: tuple):
        pass

    @abstractmethod
    def val_step_end(self):
        pass

    @abstractmethod
    def epoch_end(self):
        pass

    @abstractmethod
    def test_step(self, data: tuple):
        pass

    def load_checkpoints(self, checkpoints: str):
        state_dict = torch.load(checkpoints, map_location=self.device)
        self.model.load_state_dict(state_dict['weights'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self._last_epoch = state_dict['epoch']

    def view_metrics(self):
        for name, v in self._results['val'].items():
            if 'Metric' in name:
                print(f'{name}:\n{v}')
