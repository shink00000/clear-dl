from torch.utils.data import DataLoader

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

    def train_dataloader(self):
        return self._dataloader(phase='train')

    def val_dataloader(self):
        return self._dataloader(phase='val')

    def test_dataloader(self):
        return self._dataloader(phase='test')

    def _phase_dict(self, d: dict, phase: str):
        phase_d = {}
        for key, val in d.items():
            if isinstance(val, dict):
                if phase in val:
                    val = val[phase]
                else:
                    val = self._phase_dict(val, phase)
            phase_d[key] = val
        return phase_d

    def _dataloader(self, phase: str):
        dataset = self._phase_dict(self.dataset, phase)
        dataset = DATASETS[dataset.pop('type')](phase=phase, **dataset)

        dataloader = self._phase_dict(self.dataloader, phase)
        return DataLoader(
            dataset,
            shuffle=phase == 'train',
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            drop_last=phase == 'train',
            **dataloader
        )
