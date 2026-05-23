import math

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import ToTensorV2
from tqdm import tqdm

from datasets import get_pulse_df_train, get_pulse_df_test
from paths import TRAINING_IMAGES_DIR
from training.config import BROKEN_IMAGES_IDS
from training.embeddings.utils import get_emb_filename
from training.pretrained_models import get_backbone, EMBEDDINGS

BATCH_SIZE = 128
device = "cuda"
DIM = 512


def load_image(path, transform):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(image=img)["image"]


if __name__ == "__main__":

    models = {
        name: get_backbone(name).to(device).eval()
        for name in EMBEDDINGS
    }
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])

    train_df = get_pulse_df_train()
    test_df = get_pulse_df_test()

    train_df = train_df[~(
            train_df["place_id_left_new"].isin(BROKEN_IMAGES_IDS) |
            train_df["place_id_right_new"].isin(BROKEN_IMAGES_IDS)
    )]

    test_df = test_df[~(
            test_df["place_id_left_new"].isin(BROKEN_IMAGES_IDS) |
            test_df["place_id_right_new"].isin(BROKEN_IMAGES_IDS)
    )]

    train_df = train_df.assign(split_flag=0)
    test_df = test_df.assign(split_flag=1)
    train_size = len(train_df)
    test_size = len(test_df)

    full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    place_left = full_df["place_name_left"].to_numpy()
    place_right = full_df["place_name_right"].to_numpy()
    hv_left = full_df["place_id_left_new"].to_numpy()
    hv_right = full_df["place_id_right_new"].to_numpy()

    paths = set()

    for pl, hv in zip(place_left, hv_left):
        paths.add((hv, str(TRAINING_IMAGES_DIR / pl / f"{hv}.png")))

    for pl, hv in zip(place_right, hv_right):
        paths.add((hv, str(TRAINING_IMAGES_DIR / pl / f"{hv}.png")))

    hv_paths = sorted(list(paths))

    hv_to_idx = {hv: i for i, (hv, _) in enumerate(hv_paths)}

    train_maps_left = {}
    train_maps_right = {}
    test_maps_left = {}
    test_maps_right = {}

    for name in models:
        train_maps_left[name] = np.memmap(
            get_emb_filename("left", name, True),
            dtype="float32",
            mode="w+",
            shape=(train_size, DIM),
        )

        train_maps_right[name] = np.memmap(
            get_emb_filename("right", name, True),
            dtype="float32",
            mode="w+",
            shape=(train_size, DIM),
        )

        test_maps_left[name] = np.memmap(
            get_emb_filename("left", name, False),
            dtype="float32",
            mode="w+",
            shape=(test_size, DIM),
        )

        test_maps_right[name] = np.memmap(
            get_emb_filename("right", name, False),
            dtype="float32",
            mode="w+",
            shape=(test_size, DIM),
        )

    n_batches = math.ceil(len(hv_paths) / BATCH_SIZE)

    emb_buffer = {name: np.empty((len(hv_paths), DIM), dtype=np.float32) for name in models}
    with torch.inference_mode():

        for batch_idx in tqdm(range(n_batches)):
            start = batch_idx * BATCH_SIZE
            end = min(len(hv_paths), start + BATCH_SIZE)

            imgs = [
                load_image(hv_paths[i][1], transform)
                for i in range(start, end)
            ]

            x = torch.stack(imgs, dim=0).to(device)

            for name, model in models.items():
                feat = model(x).cpu().numpy()
                emb_buffer[name][start:end] = feat
    train_i = 0
    test_i = 0
    for i, row in enumerate(tqdm(full_df.itertuples(index=False), total=len(full_df))):
        li = hv_to_idx[row.place_id_left_new]
        ri = hv_to_idx[row.place_id_right_new]

        is_train = row.split_flag == 0
        for name in models:

            if is_train:
                train_maps_left[name][train_i] = emb_buffer[name][li]
                train_maps_right[name][train_i] = emb_buffer[name][ri]
            else:
                test_maps_left[name][test_i] = emb_buffer[name][li]
                test_maps_right[name][test_i] = emb_buffer[name][ri]
        if is_train:
            train_i += 1
        else:
            test_i += 1

    for d in [train_maps_left, train_maps_right, test_maps_left, test_maps_right]:
        for m in d.values():
            m.flush()
