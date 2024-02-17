__all__ = ["UnzippedDatasets", "DefaultUnzippedDatasets"]
import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from utils.zip import unzip_file


@dataclass
class UnzippedDatasets:

    def __init__(self, gt_root: Path):
        self.gt_root = gt_root

    @property
    def img_root(self) -> Path:
        return self.gt_root / "leftImg8bit"

    @property
    def img_root_train(self) -> Path:
        return self.img_root / "train"

    @property
    def img_root_val(self) -> Path:
        return self.img_root / "val"

    @property
    def img_root_test(self) -> Path:
        return self.img_root / "test"

    @property
    def stamp_file(self) -> Path:
        return self.gt_root / "data.stamp"

    @property
    def img_root_exists(self) -> bool:
        if self.img_root.exists():
            return True
        return False

    def check_Img8bit_file_structure(self, split: str):
        images_subfolder = self.img_root / split
        if not images_subfolder.exists():
            any_images_in_folder = [p for p in sorted(self.img_root.glob(f"**/*{split}")) if p.is_dir()]
            if len(any_images_in_folder) == 0:
                raise RuntimeError(f"No {split} folders found in subfolders of '{self.img_root}'.")
            elif len(any_images_in_folder) > 1:
                raise RuntimeError(f"More than 1 {split} folder found in subfolders of '{self.img_root}'.")
            shutil.move(any_images_in_folder[0], images_subfolder)

            # check for any empty subfolders of self.img_root after the above move and delete them.
            dirs = [p for p in self.img_root.glob("*") if p.is_dir()]
            for dir in dirs:
                if not any(dir.iterdir()):
                    shutil.rmtree(dir)

    def check_Img8bit_train(self):
        if not self.img_root_train.exists():
            self.check_Img8bit_file_structure("train")

    def check_Img8bit_val(self):
        if not self.img_root_val.exists():
            self.check_Img8bit_file_structure("val")

    def check_Img8bit_test(self):
        if not self.img_root_test.exists():
            self.check_Img8bit_file_structure("test")


DataPath = Path("data")
gtFine_trainvaltest_zip_path = DataPath / "gtFine_trainvaltest.zip"
gtFine_trainvaltest_path = DataPath / "gtFine_trainvaltest"
leftImg8bit_trainvaltest_zip_path = DataPath / "leftImg8bit_trainvaltest.zip"
DatasetStampFilePath = Path("data/data.stamp")


DefaultUnzippedDatasets: UnzippedDatasets = UnzippedDatasets(gtFine_trainvaltest_path)


def check_dataset_stamp_file(root_path: Optional[Path] = None, verbose: bool = False) -> bool:
    if root_path is None:
        stamp_path = DefaultUnzippedDatasets.stamp_file
    elif root_path.is_dir():
        stamp_path = UnzippedDatasets(root_path).stamp_file
    else:
        stamp_path = None

    if stamp_path is not None:
        if stamp_path.exists():
            with open(str(stamp_path), 'r') as f:
                lines = [l.strip("\n") for l in f.readlines()]
            if len(lines) > 0:
                if lines[0] in ["Successfully unzipped the data"]:
                    if verbose:
                        print(f"'{stamp_path}' indicates data has already been successfully unzipped.")
                    return True

    return False


def write_dataset_stamp_file(root_path: Optional[Path] = None):
    if root_path is None:
        stamp_path = DefaultUnzippedDatasets.stamp_file
    elif root_path.is_dir():
        stamp_path = UnzippedDatasets(root_path).stamp_file
    else:
        raise RuntimeError(f"'{root_path}' is not a directory.")

    with open(str(stamp_path), 'w') as f:
        f.write("Successfully unzipped the data")


def unzip_cityscapes_dataset(
    gt_root: Optional[Union[str, Path]] = None,
    img_root: Optional[Union[str, Path]] = None,
) -> UnzippedDatasets:
    if check_dataset_stamp_file(root_path=gt_root, verbose=True):
        unzipped_dataset = UnzippedDatasets(gt_root) if gt_root is not None else DefaultUnzippedDatasets
        stamp_file = unzipped_dataset.stamp_file
        print(f"If you wish to unzip the files again, please delete the stamp file at '{stamp_file}'.")
        return unzipped_dataset

    if gt_root is None:
        gt_root = gtFine_trainvaltest_zip_path

    gt_root = Path(gt_root)
    if not gt_root.exists():
        raise RuntimeError(f"'{gt_root}' does not exist.")

    # unzip gtFine_trainvaltest.zip
    if not gt_root.is_dir():
        dest = unzip_file(gt_root, dry_run=True)
        if not dest.exists():
            dest = unzip_file(gt_root)
            print(f"Successfully unzipped '{gt_root}' to '{dest}'")
        else:
            print(f"'{gt_root}' already unzipped to '{dest}'.")

    unzipped_dataset = UnzippedDatasets(gt_root)

    if img_root is None:
        img_root = leftImg8bit_trainvaltest_zip_path

    img_root = Path(img_root)
    if not img_root.exists():
        raise RuntimeError(f"'{img_root}' does not exist.")

    # unzip leftImg8bit_trainvaltest.zip
    if not img_root.is_dir():
        dest = unzip_file(img_root, dest=unzipped_dataset.img_root, dry_run=True)
        if not dest.exists():
            dest = unzip_file(img_root, dest=unzipped_dataset.img_root)
            print(f"Successfully unzipped '{img_root}' to '{dest}'.")
        else:
            print(f"'{img_root}' already unzipped to '{dest}'.")

    unzipped_dataset.check_Img8bit_train(), unzipped_dataset.check_Img8bit_val(), unzipped_dataset.check_Img8bit_test()

    return unzipped_dataset


def main_cli(args=None):
    parser = argparse.ArgumentParser(description="Unzip CityScapes DataSet.")
    parser.add_argument(
        "--labels-zip-path",
        "-l",
        type=str,
        required=False,
        default=None,
        help="path to labels zip file (gtFine_trainvaltest.zip)",
        dest="labels_zip_path_s",
    )
    parser.add_argument(
        "--images-zip-path",
        "-i",
        type=str,
        required=False,
        default=None,
        help="path to images zip file (leftImg8bit_trainvaltest.zip)",
        dest="images_zip_path_s",
    )
    parsed_args = parser.parse_args()
    labels_zip_path = parsed_args.labels_zip_path_s
    root_path = None
    if labels_zip_path is not None:
        labels_zip_path = Path(labels_zip_path)
        root_path = labels_zip_path.parent / labels_zip_path.stem

    images_zip_path = parsed_args.images_zip_path_s
    if images_zip_path is not None:
        images_zip_path = Path(images_zip_path)

    if check_dataset_stamp_file(root_path, verbose=True):
        print("Exiting...")
        sys.stdout.flush()
        exit(0)

    unzip_cityscapes_dataset(gt_root=labels_zip_path, img_root=images_zip_path)

    write_dataset_stamp_file(root_path)
    sys.stdout.flush()


if __name__ == '__main__':
    main_cli()
