__all__ = ["FcnResNet50BackBone"]
import time

import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from utils.torch_model import freeze_layers

from data_prep.cityscape_dataset import get_dataset, LoadedDatasets


class FcnResNet50Wrapper(nn.Module):
    def __init__(self, weights: FCN_ResNet50_Weights = FCN_ResNet50_Weights.DEFAULT, n_layers_to_not_freeze: int = 0):
        super().__init__()

        self.resnet50 = fcn_resnet50(weights=weights)
        if n_layers_to_not_freeze >= 0:
            freeze_layers(self.resnet50.backbone, n_layers_to_not_freeze)

    def forward(self, x) -> torch.Tensor:
        return self.resnet50(x)["out"]


class FcnResNet50BackBone(FcnResNet50Wrapper):
    def __init__(
        self,
        n_classes: int = 20,
        weights: FCN_ResNet50_Weights = FCN_ResNet50_Weights.DEFAULT,
        n_layers_to_not_freeze: int = 0,
    ):
        super().__init__(weights=weights, n_layers_to_not_freeze=n_layers_to_not_freeze)

        for children in self.resnet50.children():
            for child in children.children():
                pass
        in_channels = child.out_channels
        self.relu = nn.ReLU()
        self.proj_to_class_space = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x) -> torch.Tensor:
        return self.proj_to_class_space(self.relu(super().forward(x)))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    datasets: LoadedDatasets = get_dataset(batch_size=batch_size)
    train_set = datasets.train
    print(f"Using device: {device}")
    backbone = FcnResNet50BackBone(n_classes=20).to(device)
    img, label = next(iter(train_set))
    img = img.to(device)

    print("Running FCN_ResNet50")
    start_time = time.perf_counter()
    out_resnet50 = backbone(img)
    end_time = time.perf_counter()
    print(f"ResNet50 out.shape: {out_resnet50.shape}")
    print(f"Time Elapsed: {end_time - start_time}s")


if __name__ == '__main__':
    main()
