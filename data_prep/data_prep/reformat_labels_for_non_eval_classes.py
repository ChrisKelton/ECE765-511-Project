import shutil
from pathlib import Path

import PIL
import numpy as np
from tqdm import tqdm

from cityscape_dataset import get_dataset, LoadedDatasets, RemappedLabelsDict, targets_transform_ignore_non_eval
from unzip_cityscape_dataset import DefaultUnzippedDatasets

LabelOriginalCopyDir: Path = DefaultUnzippedDatasets.label_root.parent / "gtFine-original"
LabelOriginalCopyDir.mkdir(exist_ok=True, parents=True)


def remap_target_img_eval_classes(target: np.ndarray) -> np.ndarray:
    for original_val, remapped_val in RemappedLabelsDict.items():
        target[target == original_val] = remapped_val

    return target


def main():
    # no target transform so we can keep the data path
    datasets: LoadedDatasets = get_dataset(return_cityscape_objects=True)
    for split in ["train", "val", "test"]:
        print(f"Converting targets for '{split}' dataset.")
        dataset = getattr(datasets, split)
        for target_paths in tqdm(dataset.targets):
            for target_path in target_paths:
                target_path = Path(target_path)
                rel_path = target_path.relative_to(DefaultUnzippedDatasets.label_root)
                copy_target_path = LabelOriginalCopyDir / rel_path
                copy_target_path.parent.mkdir(exist_ok=True, parents=True)
                # don't want to accidentally overwrite original ground truth files in case this is run after
                # updating labels. And don't want to erroneously update already updated labels
                if not copy_target_path.exists():
                    shutil.copyfile(str(target_path), str(copy_target_path))
                    target = np.asarray(PIL.Image.open(target_path)).copy()
                    target = targets_transform_ignore_non_eval(target)
                    target = remap_target_img_eval_classes(target)
                    target = PIL.Image.fromarray(target)
                    target.save(str(target_path))


if __name__ == '__main__':
    main()
