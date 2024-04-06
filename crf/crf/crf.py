import networkx
from data_prep.cityscape_dataset import get_dataset, LoadedDatasets
from data_prep import DataPath


def main():
    datasets: LoadedDatasets = get_dataset(DataPath)
    train_set = datasets.train
    for idx, (img, label) in enumerate(train_set):
        if idx == 1:
            break


if __name__ == '__main__':
    main()
