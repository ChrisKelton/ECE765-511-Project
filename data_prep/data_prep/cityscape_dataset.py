__all__ = ["LoadedDatasets", "get_dataset"]

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union, Optional, Callable, List

import numpy as np
import torch
import torchvision.transforms
from torchvision.datasets import Cityscapes

from data_prep.unzip_cityscape_dataset import UnzippedDatasets, DefaultUnzippedDatasets


@dataclass
class LoadedDatasets:
    train: Cityscapes
    val: Cityscapes
    test: Cityscapes

    def __init__(self):
        pass


class IgnoreEvalClasses(Enum):
    unlabeled = 0
    ego_vehicle = 1
    rectification_border = 2
    out_of_roi = 3
    static = 4
    dynamic = 5
    ground = 6
    parking = 9
    rail_track = 10
    guard_rail = 14
    bridge = 15
    tunnel = 16
    polegroup = 18
    caravan = 29
    trailer = 30
    license_plate = -1


IgnoreEvalClassesIds: list[int] = [
    val.value for val in list(IgnoreEvalClasses.__members__.values())
]


def targets_transform_ignore_non_eval(
    target_img: Union[np.ndarray, torch.Tensor]
) -> torch.Tensor:
    for instance_id in IgnoreEvalClassesIds:
        target_img[target_img == instance_id] = 0

    return target_img


def get_dataset(
    gt_root: Optional[Union[str, Path]] = None,
    split: Optional[Union[str, List[str]]] = None,
    mode: str = "fine",
    target_type: str = "semantic",
    transform: Optional[Union[list[Callable], Callable]] = None,
    target_transform: Optional[Union[list[Callable], Callable]] = None,
    transforms: Optional[Union[list[Callable], Callable]] = None,
    ignore_non_eval_classes: bool = True,
) -> LoadedDatasets:
    """

    Args:
        gt_root: directory to gtFine_trainvaltest(.zip)
        split: Optional string to only load in train / val / test datasets.
        mode: for our purposes 'fine'
        target_type: 'semantic' for training semantic segmentation model
        transform: A function/transform that takes in a PIL image and returns a transformed version
        target_transform: A function/transform that takes in the target and transforms it
        transforms: A function/transform that takes input sample and its target as entry and returns a transformed version
        ignore_non_eval_classes: boolean to skip void classes (classes not used for evaluation)
            includes:
                - unlabeled
                - ego vehicle (vehicle taking pictures, front of car with car decal)
                - rectification border
                - out of roi
                - static
                - dynamic
                - ground
                - parking
                - rail track
                - guard rail
                - bridge
                - tunnel
                - polegroup
                - caravan
                - trailer
                - license plate

    Returns:

    """
    if gt_root is None:
        unzipped_dataset = DefaultUnzippedDatasets
    else:
        gt_root = Path(gt_root)
        if not gt_root.exists():
            raise RuntimeError(f"'{gt_root}' does not exist.")
        unzipped_dataset = UnzippedDatasets(gt_root)

    if not unzipped_dataset.img_root_exists:
        raise RuntimeError(f"'{unzipped_dataset.img_root}' does not exist.")

    if split is None:
        split = ["train", "val", "test"]
    if not isinstance(split, list):
        split = [split]

    if target_transform is None:
        # target_transform: list[Callable] = [torchvision.transforms.ToTensor()]
        target_transform: list[Callable] = []
    elif not isinstance(target_transform, list):
        target_transform = [target_transform]
    if ignore_non_eval_classes:
        target_transform.extend([lambda x: np.asarray(x).copy(), lambda x: targets_transform_ignore_non_eval(x)])

    target_transform = torchvision.transforms.Compose(target_transform)

    datasets: LoadedDatasets = LoadedDatasets()
    for split_ in split:
        datasets.__setattr__(
            split_,
            Cityscapes(
                root=str(gt_root),
                split=split_,
                mode=mode,
                target_type=target_type,
                transform=transform,
                target_transform=target_transform,
                transforms=transforms,
            ),
        )

    return datasets
