__all__ = ["Segmentor"]
import torch
import torch.nn as nn

from semantic_segmentation.backbone import FcnResNet50BackBone
from semantic_segmentation.crf import CrfRnn, CrfRnnConfig
from utils.torch_model import get_n_params


class Segmentor(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_crf_blocks: int = 5,  # it is not advised to increase this value
        channels: int = 3,
        gaussian_filter_kernel_sizes: tuple[tuple[int, ...], ...] = CrfRnnConfig["gaussian_filter_kernel_sizes"],
        gaussian_filter_sigmas: tuple[tuple[int, ...], ...] = CrfRnnConfig["gaussian_filter_sigmas"],
        bilateral_filter_kernel_sizes: tuple[tuple[int, ...], ...] = CrfRnnConfig["bilateral_filter_kernel_sizes"],
        bilateral_filter_sigmas: tuple[tuple[tuple[int, float], ...], ...] = CrfRnnConfig["bilateral_filter_sigmas"],
        freeze_backbone_layers: bool = True,
        apply_normalization: bool = False,
    ):
        super().__init__()

        self.backbone = FcnResNet50BackBone(
            n_layers_to_not_freeze=0 if freeze_backbone_layers else -1,
            apply_normalization=apply_normalization,
        )
        if not freeze_backbone_layers:
            self.backbone.requires_grad_(True)
        self.crf_net = CrfRnn(
            n_classes=n_classes,
            n_crf_blocks=n_crf_blocks,
            channels=channels,
            gaussian_filter_kernel_sizes=gaussian_filter_kernel_sizes,
            gaussian_filter_sigmas=gaussian_filter_sigmas,
            bilateral_filter_kernel_sizes=bilateral_filter_kernel_sizes,
            bilateral_filter_sigmas=bilateral_filter_sigmas,
        )

    def print_n_params(self):
        total_params = get_n_params(self)
        backbone_params = get_n_params(self.backbone)
        crf_net_params = get_n_params(self.crf_net)
        print(f"Total Number of Parameters: {total_params / 1000}K\n"
              f"\tBackbone Parameters: {backbone_params / 1000}K\n"
              f"\tCRF-RNN Parameters: {crf_net_params / 1000}K")

    def forward(self, x) -> torch.Tensor:
        x_unary_potentials = self.backbone(x)
        x_crf = self.crf_net(x, x_unary_potentials)

        return x_crf


def main():
    from data_prep.cityscape_dataset import get_dataset
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    datasets = get_dataset(batch_size=1)
    imgs, labels = next(iter(datasets.train))
    imgs = imgs.to(device)
    labels = labels.to(device)

    segmentor = Segmentor(n_classes=20).to(device)
    segmentor.print_n_params()
    out = segmentor(imgs)
    print(imgs.shape, out.shape)


if __name__ == '__main__':
    main()
