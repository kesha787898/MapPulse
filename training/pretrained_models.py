import timm
import torch
from torchgeo.models import ResNet18_Weights

EMBEDDINGS = ["sentinel2_rgb_seco", "imagenet", "custom"]


def get_sentinel2_rgb_seco():
    weights = ResNet18_Weights.SENTINEL2_RGB_SECO
    model = timm.create_model("resnet18", in_chans=weights.meta["in_chans"], num_classes=0)
    model.load_state_dict(weights.get_state_dict(progress=True), strict=True)
    return model


def get_imagenet():
    model = timm.create_model(
        "resnet18",
        pretrained=True,
        num_classes=0
    )
    return model


def get_custom_model():
    chkpt = torch.load(
        r"D:\projects\maps\pretraining\tb_logs\my_model\version_13\checkpoints\epoch=48-step=3871.ckpt",
        map_location="cpu")
    encoder_state_dict = {
        k.removeprefix("encoder."): v
        for k, v in chkpt["state_dict"].items()
        if k.startswith("encoder.")
    }

    model = timm.create_model("resnet18", in_chans=3, num_classes=0)
    model.load_state_dict(encoder_state_dict, strict=True)
    model.num_features = 512
    return model


def get_backbone(name):
    if name == "sentinel2_rgb_seco":
        return get_sentinel2_rgb_seco()
    elif name == "imagenet":
        return get_imagenet()
    elif name == "custom":
        return get_custom_model()

    else:
        raise ValueError(f"Unknown backbone name: {name}")


if __name__ == '__main__':
    print(get_custom_model())
