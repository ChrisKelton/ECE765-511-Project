__all__ = ["imshow", "ColorizeLabels", "generate_confusion_matrix_from_array"]
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision

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


def generate_confusion_matrix_from_array(
    confusion_mat: np.ndarray,
    class_names: list[str],
    img_out_path: Path,
    csv_out_path: Optional[Path] = None,
    normalize: bool = True,
    to_percentages: bool = True,
):
    if confusion_mat.shape[0] != confusion_mat.shape[1]:
        raise RuntimeError(f"confusion_mat must be a square matrix, got shape: '{confusion_mat.shape}'")
    if img_out_path.is_dir():
        raise RuntimeError(f"img_out_path must point to a file, not a directory: '{img_out_path}'")
    if csv_out_path is None:
        csv_out_path = img_out_path.parent / f"{img_out_path.stem}.csv"

    if normalize:
        for row in range(confusion_mat.shape[0]):
            sum_ = np.sum(confusion_mat[row, :])
            confusion_mat[row, :] /= sum_

    if to_percentages:
        confusion_mat *= 100

    df = pd.DataFrame(
        confusion_mat,
        index=class_names,
        columns=class_names,
    )
    df.to_csv(str(csv_out_path))
    generate_confusion_matrix_from_df(
        df=df,
        out_path=img_out_path,
        to_percentages=to_percentages,
    )


def generate_confusion_matrix_from_df(
    df: pd.DataFrame,
    out_path: Path,
    to_percentages: bool = True,
):
    plt.figure(figsize=(max((10, len(df) + 1)), max((7, len(df) - 3))))
    ax = sns.heatmap(df, annot=True, fmt=".2f")
    if to_percentages:
        for t in ax.texts:
            t.set_text(t.get_text() + "%")
    plt.savefig(str(out_path))
    plt.close()
