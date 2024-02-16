__all__ = ["LoadedDatasets", "get_dataset"]

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Callable, List

from torchvision.datasets import Cityscapes

from utils.zip import unzip_file


@dataclass
class LoadedDatasets:
    train: Cityscapes
    val: Cityscapes
    test: Cityscapes

    def __init__(self):
        pass


DataPath = Path("data")
gtFine_trainvaltest_path = DataPath / "gtFine_trainvaltest.zip"
leftImg8bit_trainvaltest_path = DataPath / "leftImg8bit_trainvaltest.zip"


def get_dataset(
    gt_root: Optional[Union[str, Path]] = None,
    img_root: Optional[Union[str, Path]] = None,
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
        img_root: directory to leftImg8bit_trainvaltest(.zip). This needs to be in the correct folder structure under
            'gt_root' in order for the dataset to be loaded in correctly.
        split: Optional string to only load in train / val / test datasets.
        mode: for our purposes 'fine'
        target_type: 'semantic' for training semantic segmentation model
        transform:
        target_transform:
        transforms:

    Returns:

    """
    gt_root_was_none: bool = False
    if gt_root is None:
        gt_root = gtFine_trainvaltest_path
        gt_root_was_none = True

    gt_root = Path(gt_root)
    if not gt_root.exists():
        raise RuntimeError(f"'{gt_root}' does not exist.")

    if not gt_root.is_dir():
        dest = unzip_file(gt_root, dry_run=True)
        if not dest.exists():
            dest = unzip_file(gt_root)
        gt_root = dest

    leftImg8bit_path = gt_root / "leftImg8bit"
    if img_root is not None and not gt_root_was_none:
        if gt_root_was_none:
            img_root = leftImg8bit_trainvaltest_path

        img_root = Path(img_root)
        if not img_root.exists():
            raise RuntimeError(f"'{img_root}' does not exist.")

        if not img_root.is_dir():
            dest = unzip_file(img_root, dest=leftImg8bit_path, dry_run=True)
            if not dest.exists():
                dest = unzip_file(img_root, dest=leftImg8bit_path)
            img_root = dest
    else:
        img_root = leftImg8bit_path

    def check_Img8bit(split: str):
        images_subfolder = img_root / split
        if not images_subfolder.exists():
            any_images_fold = [p for p in sorted(img_root.glob(f"**/*{split}")) if p.is_dir()]
            if len(any_images_fold) == 0:
                raise RuntimeError(f"No {split} folders found in subfolders of '{img_root}'.")
            elif len(any_images_fold) > 1:
                raise RuntimeError(f"More than 1 {split} folder found in subfolders of '{img_root}'.")
            shutil.move(any_images_fold[0], images_subfolder)

            dirs = [p for p in img_root.glob("*") if p.is_dir()]
            for dir in dirs:
                if not any(dir.iterdir()):
                    shutil.rmtree(dir)

    check_Img8bit("train"), check_Img8bit("test"), check_Img8bit("val")

    if split is None:
        split = ["train", "val", "test"]

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


def main_cli():
    data_path = Path("data")
    gtFine_trainvaltest_path = data_path / "gtFine_trainvaltest.zip"
    leftImg8bit_trainvaltest_path = data_path / "leftImg8bit_trainvaltest.zip"
    _ = get_dataset(gtFine_trainvaltest_path, leftImg8bit_trainvaltest_path)

    stamp_file = data_path / "data.stamp"
    with open(str(stamp_file), 'w') as f:
        f.write("Successfully unzipped the data")


if __name__ == '__main__':
    main_cli()
