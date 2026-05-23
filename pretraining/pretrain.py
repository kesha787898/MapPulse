import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from pretraining.dataset.PretrainDataset import get_loaders
from pretraining.SiamModelPretrained import SiamModelPretrained

if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_epoch",
        mode="min",
        save_top_k=1
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss_epoch",
        mode="min",
        patience=30
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
    train_loader, val_loader = get_loaders()
    model = SiamModelPretrained()

    trainer.fit(
        model,
        train_loader,
        val_loader
    )
