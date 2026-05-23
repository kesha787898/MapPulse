import pandas as pd
import torch
import yaml
from torchmetrics import Accuracy, Precision, Recall, AUROC
from tqdm import tqdm

from datasets import get_features_pulse
from paths import MODELS_TRAIN_CHKPTS, MODELS_EVAL_OUT
from training.SiamModelTrain import SiamModelTrain, MetricHolder
from training.train_dataset import get_loaders

versions = [i for i in range(24, 36)]


def get_model(path):
    chkpt = torch.load(
        path,
        map_location="cpu")
    state_dict = chkpt["state_dict"]
    model = SiamModelTrain()
    model.load_state_dict(state_dict, strict=True)
    return model


if __name__ == '__main__':
    res = []
    for version in tqdm(versions):
        d_model = {}
        path = MODELS_TRAIN_CHKPTS / f"version_{version}"
        model_file = list(path.glob("checkpoints/*.ckpt"))[0]
        num_epochs = int(model_file.stem.split("-")[0].split("=")[1])
        acc = MetricHolder(Accuracy, "accuracy", "cuda")  # TODO
        precision = MetricHolder(Precision, "precision", "cuda")
        recall = MetricHolder(Recall, "recall", "cuda")
        auroc = MetricHolder(AUROC, "auroc", "cuda")
        d_model["version"] = version
        d_model["num_epochs"] = num_epochs
        with open(path / "hparams.yaml", "r") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader)
            encoder_name = hparams["encoder_name"]
            d_model["encoder_name"] = encoder_name
            d_model["subsample_train_ratio"] = hparams["subsample_train_ratio"]
            model = get_model(model_file).to("cuda").eval()
            _, val_loader = get_loaders(encoder_name, subsample_train_ratio=None, num_workers=0)
            with torch.no_grad():
                for batch in val_loader:
                    img_1, img_2, y, mask = batch
                    img_1 = img_1.float().to("cuda")
                    img_2 = img_2.float().to("cuda")
                    y = y.float().to("cuda")
                    idx = torch.arange(mask.shape[0], device=mask.device)
                    cls = mask.argmax(dim=1)

                    y = y.float()
                    y = y
                    preds = model(img_1, img_2)[idx, cls]
                    probs = torch.sigmoid(preds)

                    acc.update_test(probs, y, cls)
                    precision.update_test(probs, y, cls)
                    recall.update_test(probs, y, cls)
                    auroc.update_test(probs, y, cls)

            for metric, name in [(acc, "acc"), (precision, "precision"), (recall, "recall"), (auroc, "auroc")]:
                metric_val = metric.compute_test()
                for val, class_name in zip(metric_val, get_features_pulse()):
                    d_model[f"{name}_{class_name}"] = val.item()
                d_model[f"{name}_mean"] = metric.compute_test_mean().item()
            res.append(d_model)
    pd.DataFrame(res).to_csv(MODELS_EVAL_OUT / "res_train.csv")
