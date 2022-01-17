import torch
from torch.utils.data import Dataset, DataLoader, random_split


import os
from glob import glob
import numpy as np
from PIL import Image

from utils import rgb2mask


class MyDataset(Dataset):
    def __init__(self, path, transform = None):
        self.path_images = glob(os.path.join(path, 'images', '*.png'))
        self.path_masks = glob(os.path.join(path, 'masks', '*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, index):
        image = Image.open(self.path_images[index])
        mask = Image.open(self.path_masks[index])

        image = np.array(image)/255
        mask = rgb2mask(np.array(mask))

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        sample = {
            'image': torch.from_numpy(image).permute(2,0,1).float(),
            'mask': torch.from_numpy(mask).long()
        }

        return sample

def make_dataloaders(path, TrainTransform, ValTransform, batch_size, num_workers):
    train_dataset = MyDataset(path + '/train', transform = TrainTransform)
    val_dataset = MyDataset(path + '/val', transform = ValTransform)
    train_loader = DataLoader(train_dataset, batch_size, num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size, num_workers, drop_last=True)

    return train_loader, val_loader