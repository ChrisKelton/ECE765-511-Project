__all__ = ["imshow", "ColorizeLabels"]
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torchvision
from dataclasses import dataclass
from data_prep.cityscape_dataset import RemappedLabels


def imshow(inp: torch.Tensor, nrows: int = 4, title: Optional[str] = None, out_path: Optional[Path] = None):
    """Display image for Tensor"""
    grid_img = torchvision.utils.make_grid(inp, nrows=nrows)
    plt.imshow(grid_img.permute(1, 2, 0))
    if title is not None:
        plt.title(title)
    if out_path is None:
        plt.show()
        input("Press Enter to continue...")
    else:
        plt.savefig(str(out_path))
    plt.close()


@dataclass
class ColorizeLabels:
    unlabeled = (0, 0, 0)
    road = (128, 64, 128)
    sidewalk = (244, 35, 232)
    building = (70, 70, 70)
    wall = (102, 102, 156)
    fence = (190, 153, 153)
    pole = (153, 153, 153)
    traffic_light = (250, 170, 30)
    traffic_sign = (220, 220, 0)
    vegetation = (107, 142, 35)
    terrain = (152, 251, 152)
    sky = (70, 130, 180)
    person = (220, 20, 60)
    rider = (255, 0, 0)
    car = (0, 0, 142)
    truck = (0, 0, 70)
    bus = (0, 60, 100)
    train = (0, 80, 100)
    motorcycle = (0, 0, 230)
    bicycle = (119, 11, 32)
    color_map_: dict[int, tuple[int, int, int]] = None

    @classmethod
    def color_map(cls) -> dict[int, tuple[int, int, int]]:
        if cls.color_map_ is None:
            color_map_: dict[int, tuple[int, int, int]] = {}
            for label, name in RemappedLabels.to_dict(inv=True).items():
                color_map_[label] = cls.__getattribute__(cls, name)
            cls.color_map_ = color_map_.copy()
        return cls.color_map_

    @classmethod
    def colorize_labels(cls, labels: torch.Tensor):
        """

        Args:
            labels: size [Batch size, 1, Height, Width]

        Returns: [Batch size, 3, Height, Width]

        """
        colored_labels: torch.Tensor = torch.zeros((labels.size(0), 3, labels.size(2), labels.size(3)))
        for label_idx, color in cls.color_map().items():
            for batch in range(colored_labels.size(0)):
                batch_labels = labels[batch].squeeze()
                batch_labels = torch.where(batch_labels == label_idx)
                colored_labels[batch, :, batch_labels[0], batch_labels[1]] = torch.Tensor(color).unsqueeze(1)

        return colored_labels
