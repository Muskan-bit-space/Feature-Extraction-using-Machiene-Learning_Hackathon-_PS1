# utils.py

import os
import cv2
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths = sorted(os.listdir(img_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.img_paths[idx]))
        mask = cv2.imread(os.path.join(self.mask_dir, self.mask_paths[idx]), 0)

        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        mask = torch.tensor(mask).long()

        return img, mask