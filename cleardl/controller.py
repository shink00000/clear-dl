import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm


class EpochLogger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir

    def open(self):
        self.train_logger = SummaryWriter(f'{self.out_dir}/train/')
        self.val_logger = SummaryWriter(f'{self.out_dir}/val/')

        self.text_logger = logging.getLogger(__name__)
        self.text_logger.setLevel(logging.DEBUG)

        file_path = f'{self.out_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        fh = logging.FileHandler(file_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(message)s'))
        self.text_logger.addHandler(fh)

    def update(self, results: dict, epoch: int):
        for phase in ['train', 'val']:
            logger = {
                'train': self.train_logger,
                'val': self.val_logger
            }[phase]
            for name, val in results[phase].items():
                if 'text' in name:
                    text = f'[Epoch {epoch}][{name}]:\n{val}'
                    self.text_logger.info(text)
                else:
                    logger.add_scalar(name, val, epoch)

    def save_checkpoints(self, checkpoints: dict, epoch: int):
        checkpoints['epoch'] = epoch
        torch.save(checkpoints, f'{self.out_dir}/latest_checkpoints.pth')

    def close(self):
        self.train_logger.close()
        self.val_logger.close()


class Controller:
    def __init__(self, module, datamodule, out_dir: str, epochs: int, eval_interval: int = 10,
                 resume_from: str = None, load_from: str = None):
        self.module = module
        self.datamodule = datamodule
        self.logger = EpochLogger(out_dir)
        self.epochs = epochs
        self.eval_interval = eval_interval
        if resume_from is not None:
            self.module.load_checkpoints(resume_from)
        elif load_from is not None:
            self.module.load_checkpoints(load_from, weights_only=True)
        self.start_epoch = self.module.last_epoch + 1

    def __call__(self, mode: str):
        self.module.info()

        assert mode in ('train', 'evaluate', 'inference')
        getattr(self, mode)()

    def train(self):
        self.logger.open()

        train_dataloader = self.datamodule.train_dataloader()
        val_dataloader = self.datamodule.val_dataloader()
        for epoch in range(self.start_epoch, self.epochs + 1):
            # epoch start
            self.module.epoch_start()

            # train
            self.module.model.train()
            for data in tqdm(train_dataloader, desc='train'):
                self.module.train_step(data)
            self.module.train_step_end()

            # validate
            self.module.model.eval()
            self.module.run_eval = (epoch % self.eval_interval == 0) or (epoch == self.epochs)
            with torch.no_grad():
                for data in tqdm(val_dataloader, desc='val'):
                    self.module.val_step(data)
                self.module.val_step_end()

            # epoch end
            self.module.epoch_end()
            self.logger.update(self.module.results, epoch)
            self.logger.save_checkpoints(self.module.checkpoints, epoch)
            print(f'[Epoch {epoch}] train_loss: {self.module.train_loss:.04f}, val_loss: {self.module.val_loss:.04f}')

        self.logger.close()

    def evaluate(self):
        self.module.model.eval()
        self.module.run_eval = True
        self.module.classwise_eval_(True)
        val_dataloader = self.datamodule.val_dataloader()
        with torch.no_grad():
            for data in tqdm(val_dataloader):
                self.module.val_step(data)
            self.module.val_step_end()
        self.module.view_metrics()

    def inference(self):
        test_dataloader = self.datamodule.test_dataloader()
        for data in tqdm(test_dataloader):
            self.module.test_step(data)
