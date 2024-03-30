import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from torchvision.datasets import Cityscapes
from tqdm import tqdm

from data_prep import gtFine_trainvaltest_path, DataPath
from data_prep.cityscape_dataset import LoadedDatasets, get_dataset


@dataclass
class PixelWiseClassPercentages:
    class_ids: list[int]
    class_names: list[str]
    class_counts: dict[str, int]
    ignore_non_eval_class: bool

    @classmethod
    def from_class_counts(
        cls, class_map: dict[str, int], class_ids: list[int], ignore_non_eval_class: bool = False,
    ) -> "PixelWiseClassPercentages":
        class_names = list(class_map.keys())
        return cls(
            class_ids=class_ids,
            class_names=class_names,
            class_counts=class_map,
            ignore_non_eval_class=ignore_non_eval_class,
        )

    @property
    def class_percentages(self) -> dict[str, float]:
        class_counts = self.class_counts.copy()
        if self.ignore_non_eval_class:
            class_counts.pop("unlabeled")
        vals = list(self.class_counts.values())
        percents = (vals / np.sum(vals)) * 100
        return {key: val for key, val in zip(list(self.class_counts.keys()), percents)}

    @property
    def max_class(self) -> Optional[tuple[str, float]]:
        class_percents = self.class_percentages
        vals = list(class_percents.values())
        if len(vals) == 0:
            return None
        idx = vals.index(max(vals))
        return list(class_percents.keys())[idx], max(vals)

    def __str__(self) -> str:
        str_ = ""
        for key, val in self.class_percentages.items():
            str_ += f"{key}: {val:.3f}% \n"

        return str_

    @property
    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=["Instance", "Id", "Pixel Percentage"])
        for class_id, (class_name, class_percent) in zip(
            self.class_ids, self.class_percentages.items()
        ):
            df.loc[len(df)] = [class_name, class_id, class_percent]

        df.set_index("Instance", inplace=True)
        df.sort_values("Id", inplace=True)
        return df


@dataclass
class ClassLabelsDict(dict):
    def __add__(self, other) -> "ClassLabelsDict":
        for key, val in other.items():
            if self.get(key) is not None:
                self[key] += val
            else:
                self[key] = val

        return self

    def __iadd__(self, other):
        return self.__add__(other)

    def __str__(self) -> str:
        temp = self.as_dict()
        return f"{temp}"

    def as_dict(self) -> dict:
        return {key: val for key, val in self.items()}


def get_class_map(cityscape_class_labels: list, key: str) -> dict:
    class_map: dict = {}
    for class_label in cityscape_class_labels:
        key_val = getattr(class_label, key)
        if key_val in list(class_map.keys()):
            raise RuntimeError(f"Non unique key found: '{key_val}'")
        class_map[key_val] = class_label

    return class_map


def class_percentages(
    gt_root: Path,
    split: Optional[Union[str, list[str]]] = None,
    base_out_path: Optional[Path] = None,
    ignore_non_eval_classes: bool = True,
):
    """

    Args:
        gt_root: root path to CityScapes unzipped dataset, corresponds to 'gtFine_trainvaltest_path' variable
        split: Optional string to specify 'train', 'val' or 'test' datasets to be analyzed
        base_out_path: Optional base output path pointing to a directory where xlsx files will be saved to
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
    if base_out_path is None:
        base_out_path = DataPath

    # get CityScape datasets
    datasets: LoadedDatasets = get_dataset(gt_root, split=split, ignore_non_eval_classes=ignore_non_eval_classes)

    if split is None:
        split = ["train", "val", "test"]
    elif not isinstance(split, list):
        split = [split]
    dataset_types: list[str] = split

    # class counts (pixel-wise) per dataset summed over all target images (labels):
    #   e.g.,
    #   { 'train': {
    #       'road': 13.524,
    #       'car': 32.421,
    #       ...
    #       },
    #     'val': {
    #       'road': 12.123,
    #       ...
    #       },
    #     'test': {
    #       ...
    #       },
    #   }
    class_counts_per_dataset: dict[str, PixelWiseClassPercentages] = {}

    # same structure as 'class_counts_per_dataset' but tracks maximum class label percentage per target image
    max_class_labels_per_dataset: dict[str, dict[str, int]] = {}

    # same structure as 'class_counts_per_dataset' but tracks how many images contain each class label
    labels_in_target_per_dataset: dict[str, dict[str, int]] = {}

    # dataframe with overall statistics per dataset
    dfs: dict[str, pd.DataFrame] = {}
    for dataset_type in dataset_types:
        # get split type dataset
        dataset: Cityscapes = getattr(datasets, dataset_type)

        # organize class labels by some unique identifier, in this case id
        # returns a dict where each key is the instance id and the resulting CityscapesClass data structure containing
        # information about each class
        class_map = get_class_map(dataset.classes, key="id")

        # mapping to retrieve class name from instance id
        class_name_map: dict[int, str] = {
            key: label.name for key, label in class_map.items()
        }

        # custom dict structure to add dicts together, helpful for updating overall pixel wise labels
        class_counts: ClassLabelsDict[str, int] = ClassLabelsDict()

        # list to keep track of class labels by their instance id (since as humans it's easier to read class names
        # but also helpful to map the class names back to their instance ids when performing training/inference)
        class_ids: list[int] = []

        # number of occurrences where some class label is the majority of the labelled pixels in an image
        max_class_labels: dict[str, int] = {}

        # number of images where a class label occurs
        labels_in_target: dict[str, int] = {}
        for _, target_img in tqdm(dataset, desc="Pixel Wise Class Percentages"):
            # each pixel contains a value with its corresponding label, get all unique values and counts to count
            # up pixel wise occurrences
            unique_vals, cnts = np.unique(target_img, return_counts=True)

            # class labels counts by pixel (comes from 'cnts' variable above)
            target_class_counts: ClassLabelsDict[str, int] = ClassLabelsDict()

            # keep track of instance ids per target as they come up in order to properly map them back
            target_ids: list[int] = []
            for unique_val, cnt in zip(unique_vals, cnts):
                # defaults non-existing keys into dict as 0
                target_class_counts.setdefault(class_name_map[unique_val], 0)
                labels_in_target.setdefault(class_name_map[unique_val], 0)
                if unique_val not in class_ids:
                    class_ids.append(unique_val)
                    target_ids.append(unique_val)

                # add pixel-wise counts
                target_class_counts[class_name_map[unique_val]] += cnt
                # for each class label, increase count existence in target image(s)
                labels_in_target[class_name_map[unique_val]] += 1

            # keeps track of class counts over the entire dataset split
            class_counts += target_class_counts

            # instantiate PixelWiseClassPercentages class to find out what label occurs the most frequently in
            # the specific target image
            target_pixel_wise_class_percentages = (
                PixelWiseClassPercentages.from_class_counts(
                    class_map=target_class_counts.copy(),
                    class_ids=target_ids.copy(),
                    ignore_non_eval_class=ignore_non_eval_classes,
                )
            )
            max_class_label = target_pixel_wise_class_percentages.max_class
            if max_class_label is not None:
                max_class_labels.setdefault(max_class_label[0], 0)
                max_class_labels[max_class_label[0]] += 1

        # keep track of stats per dataset split
        max_class_labels_per_dataset[dataset_type] = max_class_labels.copy()
        labels_in_target_per_dataset[dataset_type] = labels_in_target.copy()
        class_counts_per_dataset[
            dataset_type
        ] = PixelWiseClassPercentages.from_class_counts(
            class_map=class_counts.copy(),
            class_ids=class_ids,
            ignore_non_eval_class=ignore_non_eval_classes,
        )
        df = class_counts_per_dataset[dataset_type].to_df

        # include 'Majority Label' & 'Num Targets' to dataframe
        df["Majority Label"] = 0
        for key, val in max_class_labels_per_dataset[dataset_type].items():
            df.loc[key, "Majority Label"] = val

        df["Num Targets"] = 0
        for key, val in labels_in_target_per_dataset[dataset_type].items():
            df.loc[key, "Num Targets"] = val

        dfs[dataset_type] = df.copy(deep=True)

    # print out total stats, prettily
    for dataset_type, class_counts in class_counts_per_dataset.items():
        print(f"{dataset_type}:\n" f"{str(class_counts)}")

    out_path = base_out_path / "data-analysis-sorted-by-id.xlsx"
    if out_path.exists():
        os.remove(str(out_path))

    with pd.ExcelWriter(str(out_path), mode="w") as xlsx_writer:
        for sheet_name, df in dfs.items():
            df.to_excel(xlsx_writer, sheet_name=sheet_name, index=True)

    out_path = base_out_path / "data-analysis-sorted-by-percent.xlsx"
    if out_path.exists():
        os.remove(str(out_path))

    with pd.ExcelWriter(str(out_path), mode="w") as xlsx_writer:
        for sheet_name, df in dfs.items():
            df.sort_values("Pixel Percentage", ascending=False, inplace=True)
            df.to_excel(xlsx_writer, sheet_name=sheet_name, index=True)


def main():
    class_percentages(gtFine_trainvaltest_path, ignore_non_eval_classes=True)


if __name__ == "__main__":
    main()
