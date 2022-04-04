import torch

from .base import BaseFramework


class SemSeg(BaseFramework):
    def epoch_start(self):
        self.train_loss = 0
        self.train_counts = 0
        self.val_loss = 0
        self.val_counts = 0
        self.results = {'train': {}, 'val': {}}

        for i, lr in enumerate(sorted(set(self.scheduler.get_last_lr()))):
            self.results['train'][f'LearningRate/lr_{i}'] = lr

    def train_step(self, data: tuple):
        self._iter_counts += 1

        images, targets = data
        images, targets = images.to(self.device), targets.to(self.device)
        outputs = self.model(images)
        loss = self.model.loss(outputs, targets) / 4
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=35, norm_type=2)
        if self._iter_counts % 4 == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.train_loss += loss * images.size(0) * 4
        self.train_counts += images.size(0)
        self.scheduler.step()

    def train_step_end(self):
        self.train_loss = (self.train_loss / self.train_counts).item()
        del self.train_counts
        self.results['train']['Loss/compare'] = self.train_loss
        self.results['train']['Loss/train'] = self.train_loss

    def val_step(self, data: tuple):
        images, targets = data
        images, targets = images.to(self.device), targets.to(self.device)
        outputs = self.model(images)
        loss = self.model.loss(outputs, targets)
        self.val_loss += loss * images.size(0)
        self.val_counts += images.size(0)
        if self.run_eval:
            preds = self.model.predict(outputs)
            self.metrics.update(preds, targets)

    def val_step_end(self):
        self.val_loss = (self.val_loss / self.val_counts).item()
        del self.val_counts
        self.results['val']['Loss/compare'] = self.val_loss
        self.results['val']['Loss/val'] = self.val_loss
        if self.run_eval:
            for name, val in self.metrics.compute().items():
                if isinstance(val, dict):
                    for k, v in val.items():
                        self.results['val'][f'Metric/{name}{k}'] = v
                else:
                    self.results['val'][f'Metric/{name}'] = val
            self.metrics.reset()

    def epoch_end(self):
        self.checkpoints = {
            'weights': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

    def test_step(self, data: tuple):
        images, *_ = data
        outputs = self.model(images)
        bboxes, scores, class_ids = self.model.predict(outputs)
