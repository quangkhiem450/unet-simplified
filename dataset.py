import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

import config

class BonelevelDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        self.image_path_list = glob.glob(image_dir + "/*.jpg")
        self.mask_path_list = glob.glob(image_dir + "/*_mask.jpg")
        self.image_path_list = list(set(self.image_path_list) - set(self.mask_path_list))
        self.image_path_list.sort()
        self.mask_path_list.sort()

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        mask_path = image_path.replace(".jpg", "_mask.jpg")

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.transforms is None:
            self.transforms = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
                transforms.ToTensor()
            ])

        image = self.transforms(image)
        image = torch.cat([image, image, image], dim=0)
        mask = self.transforms(mask)

        return {"image": image, "mask": mask}