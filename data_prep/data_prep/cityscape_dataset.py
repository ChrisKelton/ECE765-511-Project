__all__ = [
    "LoadedDatasets",
    "get_dataset",
    "IgnoreEvalClasses",
    "RemappedLabels",
    "RemappedLabelsDict",
    "targets_transform_ignore_non_eval",
]

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union, Optional, Callable, List

import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes

from data_analysis import DATA_STD, DATA_MEANS
from data_prep.unzip_cityscape_dataset import UnzippedDatasets, DefaultUnzippedDatasets


class CityscapesWrapper(Cityscapes):
    def __init__(
        self,
        root: str,
        split: str = "train",
        mode: str = "fine",
        target_type: Union[list[str], str] = "instance",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        batch_size: int = 88,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            mode=mode,
            target_type=target_type,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.batch_size = batch_size


@dataclass
class LoadedDatasets:
    train: Union[Cityscapes, DataLoader]
    val: Union[Cityscapes, DataLoader]
    test: Union[Cityscapes, DataLoader]

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


class RemappedLabels(Enum):
    unlabeled = 0
    road = 1
    sidewalk = 2
    building = 3
    wall = 4
    fence = 5
    pole = 6
    traffic_light = 7
    traffic_sign = 8
    vegetation = 9
    terrain = 10
    sky = 11
    person = 12
    rider = 13
    car = 14
    truck = 15
    bus = 16
    train = 17
    motorcycle = 18
    bicycle = 19

    @classmethod
    def to_dict(cls, inv: bool = False) -> Union[dict[str, int], dict[int, str]]:
        dict_ = {
            cls.unlabeled.name: cls.unlabeled.value,
            cls.road.name: cls.road.value,
            cls.sidewalk.name: cls.sidewalk.value,
            cls.building.name: cls.building.value,
            cls.wall.name: cls.wall.value,
            cls.fence.name: cls.fence.value,
            cls.pole.name: cls.pole.value,
            cls.traffic_light.name: cls.traffic_light.value,
            cls.traffic_sign.name: cls.traffic_sign.value,
            cls.vegetation.name: cls.vegetation.value,
            cls.terrain.name: cls.terrain.value,
            cls.sky.name: cls.sky.value,
            cls.person.name: cls.person.value,
            cls.rider.name: cls.rider.value,
            cls.car.name: cls.car.value,
            cls.truck.name: cls.truck.value,
            cls.bus.name: cls.bus.value,
            cls.train.name: cls.train.value,
            cls.motorcycle.name: cls.motorcycle.value,
            cls.bicycle.name: cls.bicycle.value,
        }
        if not inv:
            return dict_

        inv_dict: dict[int, str] = {}
        for key, val in dict_.items():
            inv_dict[val] = key
        return inv_dict


RemappedLabelsDict: dict[int, int] = {
    7: RemappedLabels.road.value,
    8: RemappedLabels.sidewalk.value,
    11: RemappedLabels.building.value,
    12: RemappedLabels.wall.value,
    13: RemappedLabels.fence.value,
    17: RemappedLabels.pole.value,
    19: RemappedLabels.traffic_light.value,
    20: RemappedLabels.traffic_sign.value,
    21: RemappedLabels.vegetation.value,
    22: RemappedLabels.terrain.value,
    23: RemappedLabels.sky.value,
    24: RemappedLabels.person.value,
    25: RemappedLabels.rider.value,
    26: RemappedLabels.car.value,
    27: RemappedLabels.truck.value,
    28: RemappedLabels.bus.value,
    31: RemappedLabels.train.value,
    32: RemappedLabels.motorcycle.value,
    33: RemappedLabels.bicycle.value,
}


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
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    *,
    ignore_non_eval_classes: bool = False,
    return_cityscape_objects: bool = False,
    center_crop_size: Optional[tuple[int, int]] = None,  # (648, 648)
    resize_size: Optional[tuple[int, int]] = (224, 224),
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
        batch_size: batch size for DataLoader
        num_workers: number of workers for DataLoader
        pin_memory: pin memory when running on GPU
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
        return_cityscape_objects: boolean to apply no target_transform, only used when initially remapping
            ignore_non_eval_classes to other values in the labels
        center_crop_size: size to crop images to, (int, int). Necessary due to our very large images & not
            enough computing resources. Is NOT applied when 'return_cityscape_objects' is 'True'
        resize_size: size to resize images to (int, int). Necessary due to very large images.

    Returns:

    """
    if gt_root is None:
        unzipped_dataset = DefaultUnzippedDatasets
        gt_root = unzipped_dataset.gt_root
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

    if return_cityscape_objects:
        target_transform = None
    else:
        if target_transform is None:
            # target_transform: list[Callable] = [torchvision.transforms.ToTensor()]
            target_transform: list[Callable] = []
        elif not isinstance(target_transform, list):
            target_transform = [target_transform]
        if ignore_non_eval_classes:
            target_transform.extend(
                [
                    lambda x: np.asarray(x).copy(),
                    lambda x: targets_transform_ignore_non_eval(x),
                    lambda x: torch.tensor(x)
                ]
            )
        else:
            # use PILToTensor to maintain uint8 dtype
            target_transform.append(torchvision.transforms.PILToTensor())
        if center_crop_size is not None:
            target_transform.append(torchvision.transforms.CenterCrop(size=center_crop_size))
        if resize_size is not None:
            target_transform.append(torchvision.transforms.Resize(size=resize_size, antialias=True))
        target_transform = torchvision.transforms.Compose(target_transform)

    if transform is None:
        transform: list[Callable] = [torchvision.transforms.ToTensor()]
    # transform.append(torchvision.transforms.Normalize(DATA_MEANS, DATA_STD))
    if center_crop_size is not None and not return_cityscape_objects:
        transform.append(torchvision.transforms.CenterCrop(size=center_crop_size))
    if resize_size is not None and not return_cityscape_objects:
        transform.append(torchvision.transforms.Resize(size=resize_size, antialias=True))
    transform.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform = torchvision.transforms.Compose(transform)

    # if center_crop_size is not None:
    #     transforms = torchvision.transforms.Compose([
    #         torchvision.transforms.CenterCrop(size=center_crop_size),
    #     ])

    if pin_memory is None:
        pin_memory = True if torch.cuda.is_available() else False

    datasets: LoadedDatasets = LoadedDatasets()
    for split_ in split:
        dataset = Cityscapes(
                root=str(gt_root),
                split=split_,
                mode=mode,
                target_type=target_type,
                transform=transform,
                target_transform=target_transform,
                transforms=transforms,
            )
        if split_ == "train":
            shuffle = True
        else:
            shuffle = False

        if return_cityscape_objects:
            datasets.__setattr__(
                split_,
                dataset,
            )
        else:
            datasets.__setattr__(
                split_,
                DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                ),
            )

    return datasets
