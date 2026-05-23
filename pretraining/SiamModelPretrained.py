import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import Accuracy, Recall, Precision, AUROC

from pretraining.dataset import PretrainDataset
import timm
from torchgeo.models import ResNet18_Weights


class MetricHolderPretrain:
    def __init__(self, constructor, metric_name, device):
        self.features = PretrainDataset.featues
        self.metric_name = metric_name

        self.metrics_by_class_train = []
        self.metrics_by_class_test = []

        for _ in self.features:
            self.metrics_by_class_train.append(constructor(task="binary").to(device))
            self.metrics_by_class_test.append(constructor(task="binary").to(device))

    def update_train(self, preds, targets):
        for i, metric in enumerate(self.metrics_by_class_train):
            metric.update(preds[:, i], targets[:, i])

    def update_test(self, preds, targets):
        for i, metric in enumerate(self.metrics_by_class_test):
            metric.update(preds[:, i], targets[:, i])

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


class SiamModelPretrained(LightningModule):
    def __init__(self, lr=1e-3, out_dim=5):
        super().__init__()

        self.save_hyperparameters()

        weights = ResNet18_Weights.SENTINEL2_RGB_SECO
        model = timm.create_model("resnet18", in_chans=weights.meta["in_chans"], num_classes=0)
        model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
        self.encoder = model
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.encoder.num_features),
            nn.Linear(self.encoder.num_features, out_dim),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.acc = MetricHolderPretrain(Accuracy, "accuracy", "cuda")  # TODO
        self.precision = MetricHolderPretrain(Precision, "precision", "cuda")
        self.recall = MetricHolderPretrain(Recall, "recall", "cuda")
        self.auroc = MetricHolderPretrain(AUROC, "auroc", "cuda")

        self.freeze_backbone()

    def freeze_backbone(self):
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        for name, param in self.encoder.named_parameters():
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True

    def forward(self, x_1, x_2):
        feats_1 = self.encoder(x_1)
        feats_2 = self.encoder(x_2)
        return self.head(feats_1 - feats_2)

    def training_step(self, batch, batch_idx):
        img_1, img_2, y_1, y_2 = batch
        img_1 = img_1.float()
        img_2 = img_2.float()

        preds = self(img_1, img_2)
        y = self.create_target(y_1, y_2)
        probs = torch.sigmoid(preds)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_loss_epoch", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.acc.update_train(probs, y)
        self.precision.update_train(probs, y)
        self.recall.update_train(probs, y)
        self.auroc.update_train(probs, y)

        return loss

    def create_target(self, y_1, y_2):
        return (y_1 > y_2).float()

    def validation_step(self, batch, batch_idx):
        img_1, img_2, y_1, y_2 = batch
        img_1 = img_1.float()
        img_2 = img_2.float()

        preds = self(img_1, img_2)
        y = self.create_target(y_1, y_2)
        loss = self.loss_fn(preds, y)
        probs = torch.sigmoid(preds)

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("val_loss_epoch", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.acc.update_test(probs, y)
        self.precision.update_test(probs, y)
        self.recall.update_test(probs, y)
        self.auroc.update_test(probs, y)

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
        optimizer = torch.optim.AdamW([
            {"params": self.head.parameters(), "lr": 1e-3},
            {"params": filter(lambda p: p.requires_grad, self.encoder.parameters()), "lr": 1e-5},
        ], weight_decay=1e-4)

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
