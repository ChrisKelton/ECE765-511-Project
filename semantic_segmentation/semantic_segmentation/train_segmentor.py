__all__ = ["train", "train_one_epoch"]
import os
from pathlib import Path
from typing import Optional

import jsonpickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_prep.cityscape_dataset import get_dataset, LoadedDatasets
from semantic_segmentation.segmentor import Segmentor
from utils import SystemPath
from visualization.models import plot_vals

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CheckpointPath: Path = SystemPath / "semantic_segmentation/semantic_segmentation/train/checkpoints"


def accuracy_score(true: torch.Tensor, pred: torch.Tensor) -> float:
    diff = true - pred
    correct = torch.sum(diff == 0)
    incorrect = torch.sum(diff != 0)

    return correct / (correct + incorrect)


@torch.no_grad()
def test_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> dict[str, float]:
    loss_vals: list[float] = []
    acc_vals: list[float] = []
    for img, label in tqdm(dataloader, desc="Testing", position=0, leave=True):
        img = img.to(device)
        label = label.to(device)
        out = model(img)

        loss: torch.Tensor = criterion(out, label.reshape(label.size(0), label.size(2), label.size(3)).long())
        loss_vals.append(float(loss.detach()))
        acc_vals.append(float(accuracy_score(label.detach(), torch.argmax(out, dim=1).detach())))

    return {"loss": np.mean(loss_vals), "acc": np.mean(acc_vals)}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: nn.Module,
) -> tuple[nn.Module, dict[str, float]]:
    loss_vals: list[float] = []
    acc_vals: list[float] = []
    for img, label in tqdm(dataloader, desc="Iterating over Dataset", position=0, leave=True):
        optimizer.zero_grad()

        img = img.to(device)
        label = label.to(device)
        out = model(img)

        loss: torch.Tensor = criterion(out, label.reshape(label.size(0), label.size(2), label.size(3)).long())
        loss.backward()
        optimizer.step()

        loss_vals.append(float(loss.detach()))
        acc_vals.append(float(accuracy_score(label.detach(), torch.argmax(out, dim=1).detach())))

    results: dict[str, float] = {
        "loss": np.mean(loss_vals),
        "acc": np.mean(acc_vals)
    }

    return model, results


def train(
    model: nn.Module,
    model_name: str,
    model_chkpt_subdir: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: nn.Module,
    base_out_path: Optional[Path] = None,
    epochs: int = 10,
    epochs_to_test_val: int = 5,
    validate_first: bool = True,
) -> tuple[nn.Module, dict[str, float], dict[str, float], dict[str, float]]:
    if base_out_path is None:
        base_out_path = CheckpointPath
    base_out_path /= model_chkpt_subdir
    base_out_path.mkdir(exist_ok=True, parents=True)

    model.train()

    validate_first_results: dict[str, float] = {}
    if validate_first:
        with torch.no_grad():
            results = test_model(model, val_loader, criterion)
            validate_first_results["loss"] = results["loss"]
            validate_first_results["acc"] = results["acc"]
        print(f"Validate First Results:\n"
              f"\tLoss: {validate_first_results['loss']:.3f}\n"
              f"\tAccuracy: {validate_first_results['acc']:.3f}")

    train_results: dict[str, list[float]] = {"loss": [], "acc": []}
    best_train_result: dict[str, float] = {}

    x_val: list[int] = []
    val_results: dict[str, list[float]] = {"loss": [], "acc": []}
    best_val_result: dict[str, float] = {}
    for epoch in tqdm(range(epochs), desc="Training Model"):
        best_model: bool = False

        model, results = train_one_epoch(model, train_loader, criterion, optimizer)
        train_results["loss"].append(results["loss"])
        train_results["acc"].append(results["acc"])

        if len(train_results["acc"]) > 1 or epoch == 0:
            if epoch == 0:
                best_model = True
                best_train_result["loss"] = train_results["loss"][-1]
                best_train_result["acc"] = train_results["acc"][-1]
            elif train_results["acc"][-1] > train_results["acc"][-2]:
                best_model = True
                best_train_result["loss"] = train_results["loss"][-1]
                best_train_result["acc"] = train_results["acc"][-1]

        if ((epoch + 1) % epochs_to_test_val == 0 or best_model) or epoch == 0:
            x_val.append(epoch + 1)
            with torch.no_grad():
                results = test_model(model, val_loader, criterion)
                val_results["loss"].append(results["loss"])
                val_results["acc"].append(results["acc"])

            if epoch == 0 or val_results["acc"][-1] > val_results["acc"][-2]:
                best_val_result["loss"] = val_results["loss"][-1]
                best_val_result["acc"] = val_results["acc"][-1]
                model_out_path = base_out_path / f"{model_name}--{epoch}.pth"
                torch.save(model.state_dict(), str(model_out_path))

            train_loss = train_results["loss"][-1]
            train_acc = train_results["acc"][-1]
            val_loss = val_results["loss"][-1]
            val_acc = val_results["acc"][-1]

            print(f"\n\nEpoch {epoch}:"
                  f"\n\ttrain_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}\n\t"
                  f"train_acc: {train_acc:.3f}, val_acc: {val_acc:.3f}")

    x_train = np.arange(1, len(train_results["loss"]) + 1)
    # x_val = np.arange(1, len(val_results["loss"]) + 1, step=epochs_to_test_val)
    loss_plot_path = base_out_path / f"{model_name}--loss.png"
    plot_vals(
        x_vals=[x_train, x_val],
        y_vals=[train_results["loss"], val_results["loss"]],
        out_path=loss_plot_path,
        title="Loss",
        xlabel="Epoch",
        ylabel="Metric",
        legends=["train", "validation"],
    )

    acc_plot_path = base_out_path / f"{model_name}--acc.png"
    plot_vals(
        x_vals=[x_train, x_val],
        y_vals=[train_results["acc"], val_results["acc"]],
        out_path=acc_plot_path,
        title="Accuracy",
        xlabel="Epoch",
        ylabel="Metric",
        legends=["train", "validation"],
    )

    optimizer_out_path = base_out_path / f"{model_name}--Adam-optimizer.json"
    optimizer_out_path.write_text(jsonpickle.dumps({"optimizer": optimizer}))

    print(f"Returning best model at '{model_out_path}'")
    model.load_state_dict(torch.load(str(model_out_path)))
    return model, best_train_result, best_val_result, validate_first_results


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
    resize_size: Optional[tuple[int, int]] = None,  # (224, 224)
):
    if n_workers < 1:
        n_workers = len(os.sched_getaffinity(0))
    print(f"Using {n_workers} workers.")
    segmentor = Segmentor(n_classes=20).to(device)
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
        model_name="CRF-RNN",
        model_chkpt_subdir="CRF-RNN",
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
    main()
