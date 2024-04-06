from cityscape_dataset import get_dataset, LoadedDatasets
from tqdm import tqdm
import time


def main():
    datasets = get_dataset()
    for split in ["train", "val", "test"]:
        dataset = getattr(datasets, split)
        start_time = time.perf_counter()
        for _ in tqdm(dataset):
            pass
        end_time = time.perf_counter()
        print(f"Time elapsed: {end_time - start_time}s")


if __name__ == '__main__':
    main()
