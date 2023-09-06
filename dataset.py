import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2


class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # Image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = image/255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        # Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        mask = np.expand_dims(mask, 0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

