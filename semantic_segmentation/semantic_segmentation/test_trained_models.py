import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm

from data_prep.cityscape_dataset import get_dataset, LoadedDatasets, RemappedLabels
from data_prep.unzip_cityscape_dataset import UnzippedDatasets, gtFine_trainvaltest_path
from semantic_segmentation.backbone import FcnResNet50BackBone
from semantic_segmentation.segmentor import Segmentor
from semantic_segmentation.train_segmentor import CheckpointPath
from visualization.data import ColorizeLabels, generate_confusion_matrix_from_array

CRF_RNN_NON_FINETUNED_BACKBONE_CHKPT_PATH: Path = CheckpointPath / "CRF-RNN/CRF-RNN--29.pth"
CRF_RNN_NON_FINETUNED_BACKBONE_OUT_PATH: Path = CheckpointPath.parent / "outputs/CRF-RNN"

CRF_RNN_FINETUNED_BACKBONE_CHKPT_PATH: Path = CheckpointPath / "CRF-RNN-finetuned-backbone-frozen-layers/CRF-RNN--finetuned-backbone--frozen-layers--29.pth"
CRF_RNN_FINETUNED_BACKBONE_OUT_PATH: Path = CheckpointPath.parent / "outputs/CRF-RNN-finetuned-backbone-frozen-layers"

FCN_RESNET_CHKPT_PATH: Path = CheckpointPath / "FcnResNet/FcnResNet--30.pth"
FCN_RESNET_OUT_PATH: Path = CheckpointPath.parent / "outputs/FcnResNet"

VISUALIZATION_OUT_PATH: Path = CheckpointPath / "Visualization.png"

LabelValPath: Path = UnzippedDatasets(gt_root=gtFine_trainvaltest_path).label_root / "val"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_and_save_outputs(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    out_path: Path,
    n_classes: int = 20,
    save_images: bool = True,
):
    """ only supports batch size of 1 """
    model.eval()
    model.to(device=device)
    confusion_mat: np.ndarray = np.zeros((n_classes, n_classes))
    for (img, label), label_path in tqdm(
        zip(dataloader, dataloader.dataset.targets),
        desc="Generating Predictions",
        total=len(dataloader),
        position=0,
        leave=True,
    ):
        y_true = torch.ravel(label).to(torch.uint8).detach().tolist()
        img = img.to(device)
        label_path: Path = Path(label_path[0])
        rel_path = label_path.relative_to(LabelValPath)
        out_path_ = out_path / rel_path
        out_path_.parent.mkdir(exist_ok=True, parents=True)
        out = torch.argmax(model(img), dim=1)
        y_pred = torch.ravel(out).to(torch.uint8).detach().tolist()
        indices = np.column_stack([y_pred, y_true])
        unique_vals, cnts = np.unique(indices, return_counts=True, axis=0)
        for unique_val, cnt in zip(unique_vals, cnts):
            confusion_mat[unique_val[0], unique_val[1]] += cnt

        if save_images:
            if len(out.shape) == 3:  # indicates batch size of 1
                out = out[None, :, :, :]
            colored_out = ColorizeLabels.colorize_labels(out).squeeze().to(torch.uint8)
            pil_out: Image = F.to_pil_image(colored_out)
            pil_out.save(str(out_path_))

    generate_confusion_matrix_from_array(
        confusion_mat=confusion_mat,
        class_names=list(RemappedLabels.to_dict().keys()),
        img_out_path=out_path / "ConfusionMatrix.png",
        csv_out_path=out_path / "confusion-mat.csv",
    )


def main():
    datasets: LoadedDatasets = get_dataset(
        batch_size=1,
        resize_size=None,
        return_cityscape_objects=False,
        num_workers=len(os.sched_getaffinity(0)),
        ignore_normalization=True,
    )
    models = [
        FcnResNet50BackBone,
        Segmentor,
        Segmentor,
    ]
    kwargs_list = [
        {"apply_normalization": True},
        {"n_classes": 20, "apply_normalization": True},
        {"n_classes": 20, "apply_normalization": True},
    ]
    models_paths = [
        FCN_RESNET_CHKPT_PATH,
        CRF_RNN_FINETUNED_BACKBONE_CHKPT_PATH,
        CRF_RNN_NON_FINETUNED_BACKBONE_CHKPT_PATH,
    ]
    out_paths = [
        FCN_RESNET_OUT_PATH,
        CRF_RNN_FINETUNED_BACKBONE_OUT_PATH,
        CRF_RNN_NON_FINETUNED_BACKBONE_OUT_PATH,
    ]
    model_names = [
        "FcnResNet50 Backbone",
        "CRF-RNN--FcnResNet-Finetuned",
        "CRF-RNN--FcnResNet-Not-Finetuned",
    ]
    save_images: bool = False
    save_visualization: bool = True
    images_to_plot: dict[str, torch.Tensor] = {}
    if save_visualization:
        img, label = datasets.val.dataset.__getitem__(0)
        colored_label = ColorizeLabels.colorize_labels(label[None, :, :, :])
        images_to_plot = {
            "original": img.clone().squeeze().permute(1, 2, 0),
            "ground-truth": colored_label.clone().squeeze().permute(1, 2, 0)
        }

    for model, kwargs, model_path, out_path, model_name in zip(models, kwargs_list, models_paths, out_paths, model_names):
        print(f"Instantiating {model_name}")
        model: nn.Module = model(**kwargs)
        model.load_state_dict(torch.load(str(model_path)))
        model.eval()
        test_and_save_outputs(
            model=model,
            dataloader=datasets.val,
            out_path=out_path,
            save_images=save_images,
        )
        if save_visualization:
            print(f"Evaluating {model_name} for visualization[")
            out = model(img.clone())
            out = ColorizeLabels.colorize_labels(torch.argmax(out, dim=1)[None, :, :, :])
            images_to_plot[model_name] = out.clone().squeeze().permute(1, 2, 0)

    if save_visualization:
        print("Plotting...")
        fig, ax = plt.subplots(nrows=1, ncols=len(images_to_plot), figsize=(10, 10))
        for idx, (key, val) in enumerate(images_to_plot.items()):
            ax[idx].imshow(val)
            ax[idx].set_title(key)

        plt.setp(ax, xticks=[], yticks=[])
        plt.tight_layout()
        plt.show()
        fig.savefig(str(VISUALIZATION_OUT_PATH))
        plt.close(fig)


if __name__ == '__main__':
    main()
