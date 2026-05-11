import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

# Proprietary VisA layout: <root>/<category>/{train,test}/{good,bad}, ground_truth/bad/*.JPG
CLASS_NAMES = [
    'candle', 'capsules', 'cashew', 'chewinggum',
    'fryum', 'macaroni1', 'macaroni2',
    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum',
]


def _list_images(dir_path):
    if not os.path.isdir(dir_path):
        return []
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')
    names = sorted(os.listdir(dir_path))
    return [os.path.join(dir_path, f) for f in names if f.lower().endswith(exts)]


class VISADataset(Dataset):
    def __init__(self, dataset_path, class_name='candle', is_train=True,
                 resize=256, cropsize=224):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        self.x, self.y, self.mask = self.load_dataset_folder()

        self.transform_x = T.Compose([T.Resize(resize, interpolation=InterpolationMode.LANCZOS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, interpolation=InterpolationMode.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask).convert('L')
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        x, y, mask = [], [], []
        base = os.path.join(self.dataset_path, self.class_name)

        if self.is_train:
            good_dir = os.path.join(base, 'train', 'good')
            paths = _list_images(good_dir)
            x.extend(paths)
            y.extend([0] * len(paths))
            mask.extend([None] * len(paths))
        else:
            gt_bad_dir = os.path.join(base, 'ground_truth', 'bad')
            for split_name, label in (('good', 0), ('bad', 1)):
                split_dir = os.path.join(base, 'test', split_name)
                img_paths = _list_images(split_dir)
                x.extend(img_paths)
                y.extend([label] * len(img_paths))
                if label == 0:
                    mask.extend([None] * len(img_paths))
                else:
                    for p in img_paths:
                        fname = os.path.basename(p)
                        stem = os.path.splitext(fname)[0]
                        # masks may share basename with different ext
                        cand = os.path.join(gt_bad_dir, fname)
                        if not os.path.isfile(cand):
                            found = None
                            for f in os.listdir(gt_bad_dir):
                                if os.path.splitext(f)[0] == stem:
                                    found = os.path.join(gt_bad_dir, f)
                                    break
                            cand = found if found else cand
                        mask.append(cand)

        assert len(x) == len(y) == len(mask), 'VisA list length mismatch'
        return list(x), list(y), list(mask)
