from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from torchvision.datasets import Cityscapes
from tqdm import tqdm

from data_prep import gtFine_trainvaltest_path
from data_prep.cityscape_dataset import LoadedDatasets, get_dataset
from dataclasses import dataclass


@dataclass
class PixelWiseClassPercentages:
    class_ids: list[int]
    class_names: list[str]
    class_counts: dict[str, int]

    @classmethod
    def from_class_counts(cls, class_map: dict[str, int], class_ids: list[int]) -> "PixelWiseClassPercentages":
        class_names = list(class_map.keys())
        return cls(
            class_ids=class_ids,
            class_names=class_names,
            class_counts=class_map,
        )

    @property
    def class_percentages(self) -> dict[str, float]:
        vals = list(self.class_counts.values())
        percents = vals / np.sum(vals)
        return {key: val for key, val in zip(list(self.class_counts.keys()), percents)}

    @property
    def max_class(self) -> tuple[str, float]:
        class_percents = self.class_percentages
        vals = list(class_percents.values())
        idx = vals.index(max(vals))
        return list(class_percents.keys())[idx], max(vals)

    def __str__(self) -> str:
        str_ = ""
        for key, val in self.class_percentages.items():
            str_ += f"{key}: {val * 100:.3f}% \n"

        return str_


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


def class_percentages(gt_root: Path, split: Optional[str] = None):
    datasets: LoadedDatasets = get_dataset(gt_root, split=split)
    dataset_types: list[str] = ["train", "val", "test"]
    class_counts_per_dataset: dict[str, PixelWiseClassPercentages] = {}
    max_class_labels_per_dataset: dict[str, dict[str, int]] = {}
    for dataset_type in dataset_types:
        dataset: Cityscapes = getattr(datasets, dataset_type)
        class_map = get_class_map(dataset.classes, key="id")
        class_name_map: dict[int, str] = {key: label.name for key, label in class_map.items()}
        class_counts: ClassLabelsDict[str, int] = ClassLabelsDict()
        class_ids: list[int] = []
        max_class_labels: dict[str, int] = {}
        for targets in tqdm(dataset.targets, desc="Pixel Wise Class Percentages"):
            for target in targets:
                target_img = np.asarray(Image.open(target))
                unique_vals, cnts = np.unique(target_img, return_counts=True)
                target_class_counts: ClassLabelsDict[str, int] = ClassLabelsDict()
                target_ids: list[int] = []
                for unique_val, cnt in zip(unique_vals, cnts):
                    target_class_counts.setdefault(class_name_map[unique_val], 0)
                    if unique_val not in class_ids:
                        class_ids.append(unique_val)
                        target_ids.append(unique_val)
                    target_class_counts[class_name_map[unique_val]] += cnt
                class_counts += target_class_counts
                target_pixel_wise_class_percentages = PixelWiseClassPercentages.from_class_counts(
                    class_map=target_class_counts.copy(),
                    class_ids=target_ids.copy()
                )
                max_class_label = target_pixel_wise_class_percentages.max_class
                max_class_labels.setdefault(max_class_label[0], 0)
                max_class_labels[max_class_label[0]] += 1

        max_class_labels_per_dataset[dataset_type] = max_class_labels.copy()

        class_counts_per_dataset[dataset_type] = PixelWiseClassPercentages.from_class_counts(
            class_map=class_counts.copy(),
            class_ids=class_ids,
        )

    for dataset_type, class_counts in class_counts_per_dataset.items():
        print(f"{dataset_type}:\n"
              f"{str(class_counts)}")


def main():
    class_percentages(gtFine_trainvaltest_path)


if __name__ == '__main__':
    main()
