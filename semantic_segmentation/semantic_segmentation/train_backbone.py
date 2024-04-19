import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

from data_prep.cityscape_dataset import get_dataset, LoadedDatasets
from semantic_segmentation.backbone import FcnResNet50BackBone
from semantic_segmentation.train_segmentor import train, test_model
from utils.torch_model import print_number_of_params

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(
    data_path: Optional[Path] = None,
    n_workers: int = -1,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    weight_decay: float = 3e-4,
    epochs: int = 100,
    epochs_to_test_val: int = 2,
    output_path: Optional[Path] = None,
    center_crop_size: Optional[tuple[int, int]] = None,
    resize_size: Optional[tuple[int, int]] = (224, 224),
):
    if n_workers < 1:
        n_workers = len(os.sched_getaffinity(0))
    print(f"Using {n_workers} workers.")
    backbone = FcnResNet50BackBone(n_classes=20).to(device=device)
    print_number_of_params(backbone)
    # freeze_layers(backbone.resnet50.backbone)
    datasets: LoadedDatasets = get_dataset(
        gt_root=data_path,
        num_workers=n_workers,
        batch_size=batch_size,
        center_crop_size=center_crop_size,
        resize_size=resize_size,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(backbone.parameters(), lr=learning_rate, weight_decay=weight_decay)
    trained_backbone, best_train_result, best_val_result, validate_first_results = train(
        model=backbone,
        model_name="FcnResNet",
        model_chkpt_subdir="FcnResNet",
        train_loader=datasets.train,
        val_loader=datasets.val,
        criterion=criterion,
        optimizer=optimizer,
        base_out_path=output_path,
        epochs=epochs,
        epochs_to_test_val=epochs_to_test_val,
    )
    test_results = test_model(trained_backbone, datasets.test, criterion)
    print(f"Best Model Results:\n"
          f"\tTrain Results: loss: {best_train_result['loss']:.3f}, acc: {(best_train_result['acc'] * 100):.3f}%\n"
          f"\tValidation Results: loss: {best_val_result['loss']:.3f}, acc: {(best_val_result['acc'] * 100):.3f}%\n"
          f"\tTest Results: loss: {test_results['loss']:.3f}, acc: {(test_results['acc'] * 100):.3f}%")


if __name__ == '__main__':
    main(batch_size=1, resize_size=None)
