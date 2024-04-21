import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from semantic_segmentation.test_trained_models import FCN_RESNET_OUT_PATH, CRF_RNN_FINETUNED_BACKBONE_OUT_PATH, \
    CRF_RNN_NON_FINETUNED_BACKBONE_OUT_PATH
from visualization.data import generate_confusion_matrix_from_df


def main():
    base_out_path = FCN_RESNET_OUT_PATH.parent
    mat_paths: list[Path] = [
        FCN_RESNET_OUT_PATH / "confusion-mat.csv",
        CRF_RNN_FINETUNED_BACKBONE_OUT_PATH / "confusion-mat.csv",
        CRF_RNN_NON_FINETUNED_BACKBONE_OUT_PATH / "confusion-mat.csv",
    ]
    model_names: list[str] = [
        "FcnResNet",
        "CRF-RNN--FcnResNet-Finetuned",
        "CRF-RNN--FcnResNet-Not-Finetuned",
    ]
    dfs: dict[str, pd.DataFrame] = {}
    for mat_path, model_name in zip(mat_paths, model_names):
        df = pd.read_csv(mat_path, header=[0], index_col=[0])
        dfs[model_name] = df.copy(deep=True)

    def print_statement_for_precision(pos_model_name: str, neg_model_name: str, trace_sum: float) -> str:
        if trace_sum > 0:
            return f"{pos_model_name} has {trace_sum:.2f}% better precision than {neg_model_name}"
        return f"{neg_model_name} has {abs(trace_sum):.2f}% better precision than {pos_model_name}"

    for model_name_combo in itertools.combinations(list(dfs.keys()), r=2):
        df = dfs[model_name_combo[0]] - dfs[model_name_combo[1]]
        generate_confusion_matrix_from_df(
            df=df,
            out_path=base_out_path / f"{model_name_combo[0]}--minus--{model_name_combo[1]}.png"
        )
        print(print_statement_for_precision(
            pos_model_name=model_name_combo[0],
            neg_model_name=model_name_combo[1],
            trace_sum=np.trace(np.array(df)),
        ))


if __name__ == '__main__':
    main()
