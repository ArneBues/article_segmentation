from torch.utils.data import DataLoader, Dataset
from PIL import Image
import config
import numpy as np
import torch


class MaskDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        image = Image.open(config.IMAGE_PATH + "/" + img['file'])
        if self.transform:
            image = self.transform(image)
        return (image, np.array([img['mask']]))
