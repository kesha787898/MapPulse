import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from datasets import get_features_pulse, get_pulse_df_train, get_pulse_df_test
from training.config import BROKEN_IMAGES_IDS
from training.embeddings.utils import get_emb_filename


class TrainDataset(Dataset):
    def __init__(self, train, embedding_type):
        df = get_pulse_df_train() if train else get_pulse_df_test()
        df = df[
            ~(
                    df["place_id_left_new"].isin(BROKEN_IMAGES_IDS) |
                    df["place_id_right_new"].isin(BROKEN_IMAGES_IDS)
            )
        ]  # TODO why they are broken????

        self.y = (df["choice"].to_numpy() == "left").astype(np.int8)

        cat2idx = {c: i for i, c in enumerate(get_features_pulse())}
        self.mask_idx = np.array(
            [cat2idx[c] for c in df["study_question"].to_numpy()],
            dtype=np.int16
        )
        self.emb_path_left = get_emb_filename("left", embedding_type, train)
        self.emb_path_right = get_emb_filename("right", embedding_type, train)

        self.embs_left = None
        self.embs_right = None

    def _init_memmap(self):
        if self.embs_left is None:
            self.embs_left = np.memmap(
                self.emb_path_left,
                dtype="float32",
                mode="r",
                shape=(len(self.y), 512)
            )

            self.embs_right = np.memmap(
                self.emb_path_right,
                dtype="float32",
                mode="r",
                shape=(len(self.y), 512)
            )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        self._init_memmap()

        y = self.y[idx]
        emb_left = self.embs_left[idx]
        emb_right = self.embs_right[idx]

        mask_idx = self.mask_idx[idx]

        mask = torch.zeros(len(get_features_pulse()))
        mask[mask_idx] = 1

        return emb_left, emb_right, y, mask


def stratified_subsample(groups, ratio, seed=42):
    rng = np.random.default_rng(seed)

    unique_groups = np.unique(groups)
    selected = []

    for g in unique_groups:
        idx = np.where(groups == g)[0]
        k = int(len(idx) * ratio)

        if k < 1 and len(idx) > 0:
            k = 1

        chosen = rng.choice(idx, size=min(k, len(idx)), replace=False)
        selected.append(chosen)

    return rng.permutation(np.concatenate(selected))


def get_loaders(embedding_type, subsample_train_ratio=None,
                            num_workers=4):
    train_ds = TrainDataset(embedding_type=embedding_type,
                            train=True,
                            )
    sampler = None
    if subsample_train_ratio and subsample_train_ratio < 1.0:
        indices = stratified_subsample(train_ds.mask_idx, subsample_train_ratio)
        sampler = SubsetRandomSampler(indices)

    test_ds = TrainDataset(embedding_type=embedding_type,
                           train=False,
                           )
    train_loader = DataLoader(
        train_ds,
        batch_size=2048,
        shuffle=None if sampler else True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        persistent_workers=True if num_workers > 0 else False,
        sampler=sampler
    )

    val_loader = DataLoader(
        test_ds,
        batch_size=2048,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        persistent_workers=True if num_workers > 0 else False,
    )
    return train_loader, val_loader
