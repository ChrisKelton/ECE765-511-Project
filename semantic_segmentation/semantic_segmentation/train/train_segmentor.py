from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_prep.cityscape_dataset import get_dataset, LoadedDatasets
from semantic_segmentation.segmentor import Segmentor
from utils import SystemPath
from utils.torch_model import print_number_of_params
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
    for img, label in tqdm(dataloader, desc="Testing"):
        img = img.to(device)
        label = label.to(device)
        out = model(img)

        loss: torch.Tensor = criterion(out, label)
        loss_vals.append(float(loss.detach()))
        pred = F.softmax(out, dim=1)
        acc_vals.append(accuracy_score(label.detach().cpu().numpy(), pred.detach().cpu().numpy()))

    return {"loss": np.mean(loss_vals), "acc": np.mean(acc_vals)}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: nn.Module,
) -> tuple[nn.Module, dict[str, float]]:
    loss_vals: list[float] = []
    acc_vals: list[float] = []
    for img, label in tqdm(dataloader, desc="Iterating over Dataset"):
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
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: nn.Module,
    base_out_path: Path = CheckpointPath,
    epochs: int = 10,
    epochs_to_test_val: int = 5,
) -> tuple[nn.Module, dict[str, float], dict[str, float]]:

    base_out_path.mkdir(exist_ok=True, parents=True)

    model.train()
    train_results: dict[str, list[float]] = {"loss": [], "acc": []}
    best_train_result: dict[str, float] = {}

    val_results: dict[str, list[float]] = {"loss": [], "acc": []}
    best_val_result: dict[str, float] = {}
    for epoch in tqdm(range(epochs), desc="Training Model"):
        best_model: bool = False

        model, results = train_one_epoch(model, train_loader, criterion, optimizer)
        train_results["loss"].append(results["loss"])
        train_results["acc"].append(results["acc"])

        if len(train_results) > 1:
            if train_results["acc"][-1] > train_results["acc"][-1]:
                best_model = True
                best_train_result["loss"] = train_results["loss"][-1]
                best_train_result["acc"] = train_results["acc"][-1]
                model_out_path = base_out_path / f"CRF-RNN--{epoch}.pth"
                torch.save(model.state_dict(), str(model_out_path))

        if epoch % (epochs_to_test_val - 1) == 0 or best_model:
            with torch.no_grad():
                results = test_model(model, val_loader, criterion)
                val_results["loss"].append(results["loss"])
                val_results["acc"].append(results["acc"])

            if best_model:
                best_val_result["loss"] = val_results["loss"][-1]
                best_val_result["acc"] = val_results["acc"][-1]

            train_loss = train_results["loss"][-1]
            train_acc = train_results["acc"][-1]
            val_loss = val_results["loss"][-1]
            val_acc = val_results["acc"][-1]

            print(f"\n\nEpoch {epoch}:"
                  f"\n\ttrain_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}\n\t"
                  f"train_acc: {train_acc:.3f}, val_acc: {val_acc:.3f}")

    x_train = np.arange(1, len(train_results["loss"]) + 1)
    x_val = np.arange(1, len(val_results["loss"]) + 1, step=epochs_to_test_val)
    loss_plot_path = base_out_path / "CRF-RNN--loss.png"
    plot_vals(
        x_vals=[x_train, x_val],
        y_vals=[train_results["loss"], val_results["loss"]],
        out_path=loss_plot_path,
        title="Loss",
        xlabel="Epoch",
        ylabel="Metric",
        legends=["train", "validation"],
    )

    acc_plot_path = base_out_path / "CRF-RNN--acc.png"
    plot_vals(
        x_vals=[x_train, x_val],
        y_vals=[train_results["acc"], val_results["acc"]],
        out_path=acc_plot_path,
        title="Accuracy",
        xlabel="Epoch",
        ylabel="Metric",
        legends=["train", "validation"],
    )

    print(f"Returning best model at '{model_out_path}'")
    return model.load_state_dict(torch.load(str(model_out_path))), best_train_result, best_val_result


def main():
    segmentor = Segmentor(n_classes=20).to(device)
    print_number_of_params(segmentor)
    batch_size = 2
    datasets: LoadedDatasets = get_dataset(batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    learning_rate: float = 1e-3
    optimizer = Adam(segmentor.parameters(), lr=learning_rate)
    epochs = 10
    epochs_to_test_val = 5
    trained_segmentor, best_train_result, best_val_result = train(
        model=segmentor,
        train_loader=datasets.train,
        val_loader=datasets.val,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        epochs_to_test_val=epochs_to_test_val,
    )
    test_results = test_model(trained_segmentor, datasets.test, criterion)
    print(f"Best Model Results:\n"
          f"\tTrain Results: loss: {best_train_result['loss']:.3f}, acc: {(best_train_result['acc'] * 100):.3f}%\n"
          f"\tValidation Results: loss: {best_val_result['loss']:.3f}, acc: {(best_val_result['acc'] * 100):.3f}%\n"
          f"\tTest Results: loss: {test_results['loss']:.3f}, acc: {(test_results['acc'] * 100):.3f}%")


if __name__ == '__main__':
    main()
