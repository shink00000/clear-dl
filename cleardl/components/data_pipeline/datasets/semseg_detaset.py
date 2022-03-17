from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision.transforms.functional import to_tensor
import numpy as np
import torch

from ..transforms import build_transforms


class SemSegDataset(Dataset):
    """
    Dataset Folder Layout
        - data_dir
            - images
                - train
                    - image_1.jpg
                    - image_2.jpg
                    ...
                - val
                    ...
            - labels
            - labelmap.json
            - train.txt
            - val.txt
    """
    collate_fn = None

    def __init__(self, phase: str, data_dir: str, transforms: dict):
        self.data_list = self._create_data_list(phase, data_dir)
        self.transforms = build_transforms(transforms[phase])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            tuple:
                image: (3, H, W)
                label: (H, W) ... HxW label map
        """
        image_path, label_path = self.data_list[idx]
        data_container = {
            'image': to_tensor(Image.open(image_path)),
            'label': torch.from_numpy(np.array(Image.open(label_path), copy=True, dtype=np.long))
        }
        data_container = self.transforms(data_container)
        image = data_container['image']
        label = data_container['label']
        return image, label

    def _create_data_list(self, phase: str, data_dir: str):
        data_list = []
        phase = 'val' if phase == 'test' else phase  # test does not always exist
        with open(Path(f'{data_dir}/{phase}.txt'), 'r') as f:
            pathlist = f.readlines()
        for line in pathlist:
            image_path, label_path = line.strip().split(' ')
            image_path = Path(data_dir) / image_path
            label_path = Path(data_dir) / label_path
            if image_path.exists() and label_path.exists():
                data_list.append((image_path, label_path))
        return data_list
