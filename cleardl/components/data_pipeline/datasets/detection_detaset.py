from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
from torchvision.transforms.functional import to_tensor
from contextlib import redirect_stdout
from os import devnull
import torch

from ..transforms import build_transforms, build_encoder


class DetectionDataset(Dataset):
    """
    Dataset Folder Layout
        - data_dir
            - annotations
                - instances_train.json (COCO Dataset format)
                - instances_val.json   (COCO Dataset format)
            - train
                - image_1.jpg
                - image_2.jpg
                ...
            - val
                ...
    """

    def __init__(self, phase: str, data_dir: str, transforms: dict, encoder: dict):
        self.data_list = self._create_data_list(phase, data_dir)
        self.transforms = build_transforms(transforms[phase])
        self.encoder = build_encoder(encoder)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            tuple:
                image:   (3, H, W)
                targets: encoded ground truth. the shape of each tensor is same as that of model outputs.
                meta:    image meta information. they are used when calculate metrics.
        """
        image_path, annos, meta = self.data_list[idx]
        data_container = {
            'image': to_tensor(Image.open(image_path)),
            'bboxes': torch.tensor(annos['bboxes']).float(),
            'labels': torch.tensor(annos['labels']).long()
        }
        data_container = self.transforms(data_container)
        image = data_container.pop('image')
        targets = self.encoder(bboxes=data_container['bboxes'], labels=data_container['labels'])
        return image, targets, meta

    def _create_data_list(self, phase: str, data_dir: str):
        data_list = []
        phase = 'val' if phase == 'test' else phase  # test does not always exist
        with redirect_stdout(open(devnull, 'w')):
            coco = COCO(data_dir + f'annotations/instances_{phase}.json')
        for image_id in coco.getImgIds():
            annos = {'bboxes': [], 'labels': []}
            for anno in coco.imgToAnns[image_id]:
                if anno['iscrowd'] == 1:
                    continue
                x, y, w, h = anno['bbox']
                annos['bboxes'].append([x, y, x+w, y+h])
                annos['labels'].append(anno['category_id'])
            if len(annos['bboxes']) == 0:
                continue
            info = coco.loadImgs(ids=[image_id])[0]
            image_path = data_dir + info['file_name']
            meta = {'image_id': image_id, 'height': info['height'], 'width': info['width']}
            data_list.append((image_path, annos, meta))
        return data_list

    @staticmethod
    def collate_fn(batch: tuple):
        images, targets, metas = zip(*batch)
        images = torch.stack(images, dim=0)
        targets = tuple(torch.stack(t, dim=0) for t in zip(*targets))

        return images, targets, metas
