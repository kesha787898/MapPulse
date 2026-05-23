import random

import albumentations as A
import cv2
import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from paths import PREPARED_TABULAR_DATASET_CSV_TEST, PREPARED_TABULAR_DATASET_CSV_TRAIN

featues = [
    "Safety Index",
    "Traffic Commute Time Index",
    "Pollution Index",
    "Culture &Environment",
    "Infrastructure"
]
TRAIN_SIZE = 20000
TEST_SIZE = 10000
SEED = 42


class PretrainDataset(Dataset):
    def __init__(self, train, transform):
        if train:
            self.df = pd.read_csv(PREPARED_TABULAR_DATASET_CSV_TRAIN)
        else:
            self.df = pd.read_csv(PREPARED_TABULAR_DATASET_CSV_TEST)
        self.train = train
        self.transform = transform

        self.paths = self.df["path"].values

        self.targets = self.df[featues].values.astype("float32")
        self.city_to_indices = self.df.groupby("City").indices
        self.cities = list(self.city_to_indices.keys())
        self.samples = []

        if not self.train:
            rng = random.Random(SEED)

            self.samples = []

            for _ in range(TEST_SIZE):
                city1, city2 = rng.sample(self.cities, 2)
                idx1 = rng.choice(list(self.city_to_indices[city1]))
                idx2 = rng.choice(list(self.city_to_indices[city2]))
                self.samples.append((idx1, idx2))

    def __len__(self):
        if self.train:
            return TRAIN_SIZE
        else:
            return TEST_SIZE

    def load_image(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img

    def sample_pair(self):
        city1, city2 = random.sample(self.cities, 2)

        idx1 = random.choice(list(self.city_to_indices[city1]))
        idx2 = random.choice(list(self.city_to_indices[city2]))

        return idx1, idx2

    def __getitem__(self, idx):
        if self.train:
            idx1, idx2 = self.sample_pair()
        else:
            idx1, idx2 = self.samples[idx]

        img1 = self.load_image(idx1)
        img2 = self.load_image(idx2)

        y1 = self.targets[idx1]
        y2 = self.targets[idx2]

        return img1, img2, y1, y2


def get_loaders():
    train_transform = A.Compose([
        A.Resize(256, 256),

        A.HorizontalFlip(p=0.5),

        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),

        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.2
        ),

        A.ImageCompression(
            quality_range=(50, 100),
            p=0.3
        ),

        A.ToGray(p=0.1),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),

        ToTensorV2(),
    ])
    test_transform = A.Compose([
        A.Resize(256, 256),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),

        ToTensorV2(),
    ])

    train_ds = PretrainDataset(
        train=True,
        transform=train_transform,
    )

    test_ds = PretrainDataset(
        train=False,
        transform=test_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader
