from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image
from tqdm import tqdm
import pickle

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

    def __init__(self, phase: str, data_dir: str, transforms: list, use_pkl: bool = False):
        self.use_pkl = use_pkl
        self.data_list = self._create_data_list(phase, data_dir)
        self.transforms = build_transforms(transforms)

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
        if self.use_pkl:
            with open(image_path, 'rb') as image_f, open(label_path, 'rb') as label_f:
                data_container = {
                    'image': pickle.load(image_f) / 255,
                    'label': pickle.load(label_f).long()
                }
        else:
            data_container = {
                'image': read_image(image_path.as_posix()) / 255,
                'label': read_image(label_path.as_posix()).squeeze().long()
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

        if self.use_pkl:
            # image data into pickle format to speed up file loading
            new_data_list = []
            for image_path, label_path in tqdm(data_list, desc='convert image into pickle'):
                image = read_image(image_path.as_posix())
                image_path = image_path.parent/f'{image_path.stem}.pkl'
                with open(image_path, 'wb') as f:
                    pickle.dump(image, f)
                label = read_image(label_path.as_posix()).squeeze()
                label_path = label_path.parent/f'{label_path.stem}.pkl'
                with open(label_path, 'wb') as f:
                    pickle.dump(label, f)
                new_data_list.append((image_path, label_path))
            data_list = new_data_list

        return data_list
