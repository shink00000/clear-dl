import torch

from .base import BaseFramework


class SemSeg(BaseFramework):
    def epoch_start(self):
        self._train_loss = 0
        self._train_counts = 0
        self._val_loss = 0
        self._val_counts = 0
        self._results = {'train': {}, 'val': {}}

        for i, lr in enumerate(sorted(set(self.scheduler.get_last_lr()))):
            self._results['train'][f'LearningRate/lr_{i}'] = lr

    def train_step(self, data: tuple):
        self.optimizer.zero_grad()
        images, targets = data
        images, targets = images.to(self.device), targets.to(self.device)
        outputs = self.model(images)
        loss = self.model.loss(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=35, norm_type=2)
        self.optimizer.step()
        self._train_loss += loss * images.size(0)
        self._train_counts += images.size(0)

    def train_step_end(self):
        self._train_loss = (self._train_loss / self._train_counts).item()
        del self._train_counts
        self._results['train']['Loss/compare'] = self._train_loss
        self._results['train']['Loss/train'] = self._train_loss
        self.scheduler.step()

    def val_step(self, data: tuple):
        images, targets = data
        images, targets = images.to(self.device), targets.to(self.device)
        outputs = self.model(images)
        loss = self.model.loss(outputs, targets)
        self._val_loss += loss * images.size(0)
        self._val_counts += images.size(0)
        if self._run_eval:
            preds = self.model.predict(outputs)
            self.metrics.update(preds, targets)

    def val_step_end(self):
        self._val_loss = (self._val_loss / self._val_counts).item()
        del self._val_counts
        self._results['val']['Loss/compare'] = self._val_loss
        self._results['val']['Loss/val'] = self._val_loss
        if self._run_eval:
            for name, val in self.metrics.compute().items():
                if isinstance(val, dict):
                    for k, v in val.items():
                        self._results['val'][f'Metric/{name}{k}'] = v
                else:
                    self._results['val'][f'Metric/{name}'] = val
            self.metrics.reset()

    def epoch_end(self):
        self._checkpoints = {
            'weights': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

    def test_step(self, data: tuple):
        images, *_ = data
        outputs = self.model(images)
        bboxes, scores, class_ids = self.model.predict(outputs)
