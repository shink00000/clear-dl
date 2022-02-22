import torch

from .base import BaseFramework


class Detection(BaseFramework):
    def epoch_start(self):
        self.train_loss = 0
        self.train_counts = 0
        self.val_loss = 0
        self.val_counts = 0
        self.results = {'train': {}, 'val': {}}

        for i, lr in enumerate(sorted(set(self.scheduler.get_last_lr()))):
            self.results['train'][f'LearningRate/lr_{i}'] = lr

    def train_step(self, data: tuple):
        self.optimizer.zero_grad()
        images, targets, *_ = data
        images, targets = images.to(self.device), tuple(t.to(self.device) for t in targets)
        # with torch.cuda.amp.autocast():
        outputs = self.model(images)
        loss = self.model.loss(outputs, targets)
        loss.backward()
        # self.scaler.scale(loss).backward()
        # self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=35, norm_type=2)
        self.optimizer.step()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        self.train_loss += loss * images.size(0)
        self.train_counts += images.size(0)

    def train_step_end(self):
        self.train_loss = (self.train_loss / self.train_counts).item()
        del self.train_counts
        self.results['train']['Loss/compare'] = self.train_loss
        self.results['train']['Loss/train'] = self.train_loss
        self.scheduler.step()

    def val_step(self, data: tuple):
        images, targets, metas = data
        images, targets = images.to(self.device), tuple(t.to(self.device) for t in targets)
        outputs = self.model(images)
        loss = self.model.loss(outputs, targets)
        self.val_loss += loss * images.size(0)
        self.val_counts += images.size(0)
        if self.run_eval:
            preds = self.model.predict(outputs)
            self.metrics.update(preds, metas)

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
