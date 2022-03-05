from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import pandas as pd

df = pd.read_csv(Path(__file__).parent/'cityscapes_labels.csv')
mapping = {id: train_id for id, train_id in zip(df['id'].values, df['trainId'].values)}

func = np.vectorize(lambda x: mapping[x].astype(np.uint8))

size_wh = (1024, 512)
data_dir = Path('./data/cityscapes')
dst_data_dir = Path('./data/cityscapes_ss')
pathlists_dir = Path('./data/cityscapes_ss/pathlists')
pathlists_dir.mkdir(exist_ok=True, parents=True)

with open('./data/cityscapes_ss/info.json', 'w') as f:
    json.dump(
        df[df['trainId'].between(0, 20)][['trainId', 'name']].set_index('trainId').to_dict()['name'],
        f, indent=4
    )

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

        image = Image.open(image_path).resize(size_wh, Image.LANCZOS)
        label = Image.open(label_path).resize(size_wh, Image.NEAREST)
        label = Image.fromarray(func(np.array(label)))

        image.save(dst_image_path)
        label.save(dst_label_path)

        dst_image_path = dst_image_path.relative_to(dst_data_dir).as_posix()
        dst_label_path = dst_label_path.relative_to(dst_data_dir).as_posix()
        dst_pathlist.append(f'{dst_image_path} {dst_label_path}\n')

    with open(pathlists_dir/f'{phase}_list.txt', 'w') as f:
        f.writelines(dst_pathlist)
