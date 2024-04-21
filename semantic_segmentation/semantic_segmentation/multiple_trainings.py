import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

from data_prep.cityscape_dataset import get_dataset, LoadedDatasets
from semantic_segmentation.segmentor import Segmentor
from semantic_segmentation.train_segmentor import train, test_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(
    model_name: str,
    data_path: Optional[Path] = None,
    pretrained_backbone_path: Optional[Path] = None,
    freeze_backbone_layers: bool = True,
    n_workers: int = -1,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    weight_decay: float = 3e-4,
    epochs: int = 100,
    epochs_to_test_val: int = 2,
    output_path: Optional[Path] = None,
    center_crop_size: Optional[tuple[int, int]] = None,
    resize_size: Optional[tuple[int, int]] = None,
):
    if n_workers < 1:
        n_workers = len(os.sched_getaffinity(0))
    print(f"Using {n_workers} workers.")
    segmentor = Segmentor(n_classes=20, freeze_backbone_layers=freeze_backbone_layers)
    if pretrained_backbone_path is not None:
        segmentor.backbone.load_state_dict(torch.load(str(pretrained_backbone_path)))
    segmentor.to(device)
    segmentor.print_n_params()
    datasets: LoadedDatasets = get_dataset(
        gt_root=data_path,
        num_workers=n_workers,
        batch_size=batch_size,
        center_crop_size=center_crop_size,
        resize_size=resize_size,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(segmentor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    trained_segmentor, best_train_result, best_val_result, validate_first_results = train(
        model=segmentor,
        model_name=model_name,
        model_chkpt_subdir=model_name,
        train_loader=datasets.train,
        val_loader=datasets.val,
        criterion=criterion,
        optimizer=optimizer,
        base_out_path=output_path,
        epochs=epochs,
        epochs_to_test_val=epochs_to_test_val,
    )
    test_results = test_model(segmentor, datasets.test, criterion)
    print(f"Best Model Results:\n"
          f"\tTrain Results: loss: {best_train_result['loss']:.3f}, acc: {(best_train_result['acc'] * 100):.3f}%\n"
          f"\tValidation Results: loss: {best_val_result['loss']:.3f}, acc: {(best_val_result['acc'] * 100):.3f}%\n"
          f"\tTest Results: loss: {test_results['loss']:.3f}, acc: {(test_results['acc'] * 100):.3f}%")


if __name__ == '__main__':
    main(
        model_name="Test",
        freeze_backbone_layers=False,
    )
