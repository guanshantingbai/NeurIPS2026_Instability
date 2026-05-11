import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CLIPDataset(Dataset):
    def __init__(
        self,
        load_function,
        category,
        phase,
        k_shot,
        img_resize: int | None = None,
        transform=None,
        load_memory: bool = False,
    ):

        self.load_function = load_function
        self.phase = phase

        self.category = category
        self.img_resize = img_resize
        self.transform = transform
        self.load_memory = load_memory
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # load datasets
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset(k_shot)  # self.labels => good : 0, anomaly : 1
        if self.load_memory:
            for i in range(len(self.img_paths)):
                _ = self._get_img_and_gt(i)

    def load_dataset(self, k_shot):

        (train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types), \
        (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types) = self.load_function(self.category,
                                                                                                      k_shot)
        if self.phase == 'train':

            return train_img_tot_paths, \
                   train_gt_tot_paths, \
                   train_tot_labels, \
                   train_tot_types
        else:
            return test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types

    def __len__(self):
        return len(self.img_paths)

    def _get_img_and_gt(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]

        img_path, gt = self.img_paths[idx], self.gt_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        # Convert BGR -> RGB once here (avoid per-batch Python loops later)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if gt == 0:
            gt_arr = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        else:
            gt_arr = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
            if gt_arr is None:
                raise FileNotFoundError(f"Failed to read mask: {gt}")
            gt_arr[gt_arr > 0] = 255

        if self.img_resize is not None:
            img = cv2.resize(img, (self.img_resize, self.img_resize), interpolation=cv2.INTER_LINEAR)
            gt_arr = cv2.resize(gt_arr, (self.img_resize, self.img_resize), interpolation=cv2.INTER_NEAREST)

        if self.load_memory:
            self._cache[idx] = (img, gt_arr)

        return img, gt_arr

    def __getitem__(self, idx):
        img_path, label, img_type = self.img_paths[idx], self.labels[idx], self.types[idx]
        img, gt = self._get_img_and_gt(idx)

        img_name = f'{self.category}-{img_type}-{os.path.basename(img_path[:-4])}'

        if self.transform is not None:
            # transform expects PIL RGB
            img = self.transform(Image.fromarray(img))

        return img, gt, label, img_name, img_type
