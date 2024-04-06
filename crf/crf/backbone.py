import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

from data_prep.cityscape_dataset import get_dataset, LoadedDatasets


class VggWrapper(nn.Module):
    def __init__(self, weights: VGG16_Weights = VGG16_Weights.DEFAULT):
        super().__init__()

        self.vgg16 = vgg16(weights=weights)

    def forward(self, x):
        return self.vgg16(x)


class FcnResNet50Wrapper(nn.Module):
    def __init__(self, weights: FCN_ResNet50_Weights = FCN_ResNet50_Weights.DEFAULT):
        super().__init__()

        self.resnet50 = fcn_resnet50(weights=weights)

    def forward(self, x):
        return self.resnet50(x)["out"]

def main():
    datasets: LoadedDatasets = get_dataset()
    train_set = datasets.train
    vgg16_wrapper = VggWrapper()
    resnet50_wrapper = FcnResNet50Wrapper()
    img, label = next(iter(train_set))

    # print("Running Vgg16")
    # out_vgg16 = vgg16_wrapper(img_batches)
    # print(f"Vgg16 out.shape: {out_vgg16.shape}")

    print("Running FCN_ResNet50")
    out_resnet50 = resnet50_wrapper(img)["out"]
    print(f"ResNet50 out.shape: {out_resnet50.shape}")
    a = 0


if __name__ == '__main__':
    main()
