import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torchvision import datasets
from torch.utils.data import DataLoader
import os


class AlbumentationsTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):

        img_np = np.array(img.convert('L'))
        img_np = np.expand_dims(img_np, axis=-1)

        augmented = self.transforms(image=img_np)
        return augmented["image"]

class TBDataLoader:
    def __init__(self, root_dir, image_size, batch_size):

        self.root_dir = root_dir
        self.batch_size = batch_size

        base_transform = [
            A.Resize(image_size, image_size),
            A.Normalize(mean=0.485, std=0.26),
            ToTensorV2()
        ]

        self.transforms_train = A.Compose(
            [

                # A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(0.05, 0.05),
                    rotate=(-5, 5),
                    shear=(-5, 5),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    # mask_mode=0,
                    p=0.5  #
                ),

                # A.GaussNoise(var_limit=(10.0, 50.0)),
                *base_transform,
            ]
        )

        self.transforms_val = A.Compose([*base_transform])

    def get_loaders(self):

        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.root_dir, "train"),
            transform=AlbumentationsTransform(self.transforms_train)
        )

        val_dataset = datasets.ImageFolder(
            root=os.path.join(self.root_dir, "val"),
            transform=AlbumentationsTransform(self.transforms_val)
        )


        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

        return train_loader, val_loader, train_dataset.class_to_idx

