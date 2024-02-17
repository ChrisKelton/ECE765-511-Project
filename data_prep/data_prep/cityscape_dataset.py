__all__ = ["LoadedDatasets", "get_dataset"]

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Callable, List

from torchvision.datasets import Cityscapes

from data_prep.unzip_cityscape_dataset import UnzippedDatasets, DefaultUnzippedDatasets


@dataclass
class LoadedDatasets:
    train: Cityscapes
    val: Cityscapes
    test: Cityscapes

    def __init__(self):
        pass


def get_dataset(
    gt_root: Optional[Union[str, Path]] = None,
    split: Optional[Union[str, List[str]]] = None,
    mode: str = "fine",
    target_type: str = "semantic",
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    transforms: Optional[Callable] = None,
) -> LoadedDatasets:
    """

    Args:
        gt_root: directory to gtFine_trainvaltest(.zip)
        split: Optional string to only load in train / val / test datasets.
        mode: for our purposes 'fine'
        target_type: 'semantic' for training semantic segmentation model
        transform:
        target_transform:
        transforms:

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
            )
        )

    return datasets
