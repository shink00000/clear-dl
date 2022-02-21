""" City Scapes

    name                   id    trainId  category        catId  hasInstances
--  -------------------  ----  ---------  ------------  -------  --------------
 0  unlabeled               0        255  void                0  False
 1  egovehicle              1        255  void                0  False
 2  rectificationborder     2        255  void                0  False
 3  outofroi                3        255  void                0  False
 4  static                  4        255  void                0  False
 5  dynamic                 5        255  void                0  False
 6  ground                  6        255  void                0  False
 7  road                    7          0  flat                1  False
 8  sidewalk                8          1  flat                1  False
 9  parking                 9        255  flat                1  False
10  railtrack              10        255  flat                1  False
11  building               11          2  construction        2  False
12  wall                   12          3  construction        2  False
13  fence                  13          4  construction        2  False
14  guardrail              14        255  construction        2  False
15  bridge                 15        255  construction        2  False
16  tunnel                 16        255  construction        2  False
17  pole                   17          5  object              3  False
18  polegroup              18        255  object              3  False
19  trafficlight           19          6  object              3  False
20  trafficsign            20          7  object              3  False
21  vegetation             21          8  nature              4  False
22  terrain                22          9  nature              4  False
23  sky                    23         10  sky                 5  False
24  person                 24         11  human               6  True
25  rider                  25         12  human               6  True
26  car                    26         13  vehicle             7  True
27  truck                  27         14  vehicle             7  True
28  bus                    28         15  vehicle             7  True
29  caravan                29        255  vehicle             7  True
30  trailer                30        255  vehicle             7  True
31  train                  31         16  vehicle             7  True
32  motorcycle             32         17  vehicle             7  True
33  bicycle                33         18  vehicle             7  True
34  licenseplate           -1         -1  vehicle             7  False
"""

from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

size_wh = (1024, 512)
id2id = np.zeros((34, 20))
id2id[:, 0] = 1
for i, j in enumerate([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33], start=1):
    id2id[j][i] = 1
    id2id[j][0] = 0

data_dir = Path('./data/cityscapes')
dst_data_dir = Path('./data/cityscapes_ss')
pathlists_dir = Path('./data/cityscapes_ss/pathlists')
pathlists_dir.mkdir(exist_ok=True, parents=True)

for phase in ['train', 'val']:
    src_image_dir = Path(f'./data/cityscapes/leftImg8bit/{phase}')
    src_label_dir = Path(f'./data/cityscapes/gtFine/{phase}')
    dst_image_dir = Path(f'./data/cityscapes_ss/images/{phase}')
    dst_label_dir = Path(f'./data/cityscapes_ss/labels/{phase}')

    image_paths = {}
    for p in src_image_dir.glob('**/*.png'):
        image_paths[p.name[:-16]] = p
    label_paths = {}
    for p in src_label_dir.glob('**/*_labelIds.png'):
        label_paths[p.name[:-20]] = p

    pathlist = []
    for key, image_path in image_paths.items():
        if key in label_paths:
            label_path = label_paths[key]
            pathlist.append((image_path, label_path))

    dst_pathlist = []
    for image_path, label_path in tqdm(pathlist):
        dst_image_path = dst_image_dir/image_path.relative_to(src_image_dir)
        dst_label_path = dst_label_dir/label_path.relative_to(src_label_dir)
        dst_image_path.parent.mkdir(exist_ok=True, parents=True)
        dst_label_path.parent.mkdir(exist_ok=True, parents=True)

        image = Image.open(image_path).resize(size_wh, Image.BILINEAR)
        label = Image.open(label_path).resize(size_wh, Image.NEAREST)
        label = Image.fromarray((np.eye(34)[np.array(label)]@id2id).argmax(axis=-1).astype(np.uint8))

        image.save(dst_image_path)
        label.save(dst_label_path)

        dst_image_path = dst_image_path.relative_to(dst_data_dir).as_posix()
        dst_label_path = dst_label_path.relative_to(dst_data_dir).as_posix()
        dst_pathlist.append(f'{dst_image_path} {dst_label_path}\n')

    with open(pathlists_dir/f'{phase}_list.txt', 'w') as f:
        f.writelines(dst_pathlist)
