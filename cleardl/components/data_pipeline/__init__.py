from torch.utils.data import DataLoader
from copy import deepcopy

from .datasets.detection_detaset import DetectionDataset
from .datasets.semseg_detaset import SemSegDataset

DATASETS = {
    'DetectionDataset': DetectionDataset,
    'SemSegDataset': SemSegDataset
}


class DataPipeline:
    def __init__(self, dataset: dict, dataloader: dict):
        self.dataset = dataset
        self.dataloader = dataloader

    def build_dataset(self, phase: str):
        if not hasattr(self, f'{phase}_dataset'):
            dataset = deepcopy(self.dataset)
            setattr(self, f'{phase}_dataset', DATASETS[dataset.pop('type')](phase=phase, **dataset))
        return getattr(self, f'{phase}_dataset')

    def train_dataloader(self):
        dataset = self.build_dataset(phase='train')
        return DataLoader(
            dataset,
            shuffle=True,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            **self.dataloader
        )

    def val_dataloader(self):
        dataset = self.build_dataset(phase='val')
        return DataLoader(
            dataset,
            shuffle=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            **self.dataloader
        )

    def test_dataloader(self):
        dataset = self.build_dataset(phase='test')
        return DataLoader(
            dataset,
            shuffle=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            **self.dataloader
        )
