import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from torchinfo import summary


class BaseFramework(nn.Module, metaclass=ABCMeta):
    def __init__(self, model: nn.Module, optimizer: object, scheduler: object, metrics: object, **kwargs):
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
        for key, val in kwargs.items():
            setattr(self, key, val)

        self.results = {'train': {}, 'val': {}}
        self.checkpoints = {}
        self.last_epoch = 0
        self.run_eval = False

        self.train_loss = 0
        self.train_counts = 0
        self.val_loss = 0
        self.val_counts = 0

    def classwise_eval_(self, classwise_eval: bool):
        for k, m in self.metrics.items():
            if hasattr(m, 'classwise'):
                m.classwise = classwise_eval
                print(f'{k} will be a classwise metric')

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

    def load_checkpoints(self, checkpoints: str, weights_only: bool = False):
        state_dict = torch.load(checkpoints, map_location=self.device)
        self.model.load_state_dict(state_dict['weights'], strict=False)
        if not weights_only:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])
            self.last_epoch = state_dict['epoch']

    def view_metrics(self):
        for name, v in self.results['val'].items():
            if 'Metric' in name:
                print(f'{name}:\n{v}')

    def info(self):
        print('=========================================================================')
        print(self)
        print('=========================================================================')
        summary(self.model, (2, 3, *self.input_size))
