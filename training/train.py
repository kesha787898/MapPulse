import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from training.SiamModelTrain import SiamModelTrain
from training.pretrained_models import EMBEDDINGS
from training.train_dataset import get_loaders

RATIOS = [0.1, 0.2, 0.3, 1]


def train(ratio, encoder_name):
    torch.set_float32_matmul_precision("high")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_epoch",
        mode="min",
        save_top_k=1
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss_epoch",
        mode="min",
        patience=20
    )
    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = Trainer(
        max_epochs=200,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stopping_callback, ],
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
        logger=logger,
        gradient_clip_val=1.0,
    )
    train_loader, val_loader = get_loaders(encoder_name, subsample_train_ratio=ratio)
    model = SiamModelTrain(encoder_name, subsample_train_ratio=ratio)

    trainer.fit(
        model,
        train_loader,
        val_loader
    )


if __name__ == '__main__':
    for ratio in RATIOS:
        for encoder_name in EMBEDDINGS:
            train(ratio, encoder_name)
