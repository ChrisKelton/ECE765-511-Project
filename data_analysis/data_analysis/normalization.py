import torch
from tqdm import tqdm

from data_prep.cityscape_dataset import get_dataset, LoadedDatasets


def get_mean_and_std() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    Calls `get_dataset` and iterates over all data to get the mean and standard deviation per image channel

    Returns: ((mean_ch0, mean_ch1, mean_ch2), (std_ch0, std_ch1, std_ch2))

    """
    datasets: LoadedDatasets = get_dataset()
    means: list[list[float]] = [[], [], []]
    stds: list[list[float]] = [[], [], []]
    for dataset_type in datasets.__annotations__.keys():
        print(f"Iterating over {dataset_type} dataset.")
        for imgs, _ in tqdm(datasets.__getattribute__(dataset_type)):
            # checking if data is not batched
            if len(imgs.shape) == 3:
                imgs = imgs.unsqueeze(0)
            for img in imgs:
                for channel in range(3):
                    means[channel].append(float(torch.mean(img[channel])))
                    stds[channel].append(float(torch.std(img[channel])))

    mean_: list[float] = []
    for mean_vals in means:
        mean_.append(sum(mean_vals) / len(mean_vals))

    std_: list[float] = []
    for std_vals in stds:
        std_.append(sum(std_vals) / len(std_vals))

    return tuple(mean_), tuple(std_)


if __name__ == '__main__':
    mean, std = get_mean_and_std()
    print(f"Mean: {mean}\n\nStd: {std}")
