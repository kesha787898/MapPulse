import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import Accuracy, Recall, Precision, AUROC

from datasets import get_features_pulse
from training.pretrained_models import get_backbone


class MetricHolder:
    def __init__(self, constructor, metric_name, device):
        self.features = get_features_pulse()
        self.metric_name = metric_name

        self.metrics_by_class_train = []
        self.metrics_by_class_test = []

        for _ in self.features:
            self.metrics_by_class_train.append(constructor(task="binary").to(device))
            self.metrics_by_class_test.append(constructor(task="binary").to(device))

    def update_train(self, preds, targets, cls):
        for i, metric in enumerate(self.metrics_by_class_train):
            mask = (cls == i)
            if mask.sum() == 0:
                continue
            metric.update(preds[mask], targets[mask])

    def update_test(self, preds, targets, cls):
        for i, metric in enumerate(self.metrics_by_class_test):
            mask = (cls == i)
            if mask.sum() == 0:
                continue
            metric.update(preds[mask], targets[mask])

    def compute_train(self):
        values = []
        for metric in self.metrics_by_class_train:
            values.append(metric.compute())
        return values

    def compute_test(self):
        values = []
        for metric in self.metrics_by_class_test:
            values.append(metric.compute())
        return values

    def compute_train_mean(self):
        vals = self.compute_train()
        return sum(vals) / len(vals)

    def compute_test_mean(self):
        vals = self.compute_test()
        return sum(vals) / len(vals)

    def reset_train(self):
        for metric in self.metrics_by_class_train:
            metric.reset()

    def reset_test(self):
        for metric in self.metrics_by_class_test:
            metric.reset()

    def log_train(self, logger, **kwargs):
        per_class = self.compute_train()
        for val, name in zip(per_class, self.features):
            logger.log(f"train/{self.metric_name}/{name}", val, **kwargs)
        logger.log(f"train/{self.metric_name}/mean", self.compute_train_mean(), **kwargs)

    def log_test(self, logger, **kwargs):

        # per-class
        per_class = self.compute_test()

        for val, name in zip(per_class, self.features):
            logger.log(f"test/{self.metric_name}/{name}", val, **kwargs)

        # mean
        logger.log(f"test/{self.metric_name}/mean", self.compute_test_mean(), **kwargs)


class SiamModelTrain(LightningModule):
    def __init__(self, encoder_name=None, lr=1e-3, out_dim=6, subsample_train_ratio=1.0):
        super().__init__()
        num_in_features = 512
        self.save_hyperparameters()
        self.pre_head = nn.Sequential(
            nn.BatchNorm1d(num_in_features),
            nn.Linear(num_in_features, num_in_features),
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(num_in_features),
            nn.Linear(num_in_features, out_dim),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.acc = MetricHolder(Accuracy, "accuracy", "cuda")  # TODO
        self.precision = MetricHolder(Precision, "precision", "cuda")
        self.recall = MetricHolder(Recall, "recall", "cuda")
        self.auroc = MetricHolder(AUROC, "auroc", "cuda")

    def forward(self, x_1, x_2):
        feats_1 = self.pre_head(x_1)
        feats_2 = self.pre_head(x_2)
        return self.head(feats_1 - feats_2)

    def training_step(self, batch, batch_idx):
        img_1, img_2, y, mask = batch
        img_1 = img_1.float()
        img_2 = img_2.float()
        idx = torch.arange(mask.shape[0], device=mask.device)
        cls = mask.argmax(dim=1)

        y = y.float()
        preds = self(img_1, img_2)[idx, cls]

        loss = self.loss_fn(preds, y)
        probs = torch.sigmoid(preds)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_loss_epoch", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.acc.update_train(probs, y, cls)
        self.precision.update_train(probs, y, cls)
        self.recall.update_train(probs, y, cls)
        self.auroc.update_train(probs, y, cls)

        return loss

    def validation_step(self, batch, batch_idx):
        img_1, img_2, y, mask = batch
        img_1 = img_1.float()
        img_2 = img_2.float()
        idx = torch.arange(mask.shape[0], device=mask.device)
        cls = mask.argmax(dim=1)

        y = y.float()
        y = y
        preds = self(img_1, img_2)[idx, cls]

        loss = self.loss_fn(preds, y)
        probs = torch.sigmoid(preds)

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("val_loss_epoch", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.acc.update_test(probs, y, cls)
        self.precision.update_test(probs, y, cls)
        self.recall.update_test(probs, y, cls)
        self.auroc.update_test(probs, y, cls)

        return loss

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)
        self.acc.log_train(logger=self, prog_bar=True, on_step=False, on_epoch=True)
        self.precision.log_train(self, prog_bar=True, on_step=False, on_epoch=True)
        self.recall.log_train(self, prog_bar=True, on_step=False, on_epoch=True)
        self.auroc.log_train(self, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.acc.log_test(logger=self, prog_bar=True, on_step=False, on_epoch=True)
        self.precision.log_test(self, prog_bar=True, on_step=False, on_epoch=True)
        self.recall.log_test(self, prog_bar=True, on_step=False, on_epoch=True)
        self.auroc.log_test(self, prog_bar=True, on_step=False, on_epoch=True)

    def on_train_epoch_start(self):
        self.acc.reset_train()
        self.precision.reset_train()
        self.recall.reset_train()
        self.auroc.reset_train()

    def on_validation_epoch_start(self):
        self.acc.reset_test()
        self.precision.reset_test()
        self.recall.reset_test()
        self.auroc.reset_test()

    def configure_optimizers(self):
        params = [
            {"params": self.head.parameters(), "lr": 1e-3},
            {"params": self.pre_head.parameters(), "lr": 1e-3},
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.9,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss_epoch"
            }
        }
