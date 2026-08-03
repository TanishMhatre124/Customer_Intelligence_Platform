"""
Load raw Olist datasets from data/raw into a dictionary of DataFrames.
"""

import pandas as pd

from src.config import RAW_FILES, raw_path


def load_raw_datasets(verbose: bool = True) -> dict:
    """
    Load every raw dataset listed in config.RAW_FILES.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of dataset name -> DataFrame.
    """
    datasets = {}

    for name in RAW_FILES:
        file_path = raw_path(name)
        try:
            datasets[name] = pd.read_csv(file_path)
            if verbose:
                print(f"Loaded {name}: {datasets[name].shape}")
        except FileNotFoundError:
            if verbose:
                print(f"File not found: {file_path}")

    return datasets


def dataset_summary(datasets: dict) -> None:
    """Print a quick shape/columns summary for each loaded dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    for name, df in datasets.items():
        print(f"\n{name.upper()}")
        print("-" * 40)
        print(f"Rows    : {df.shape[0]}")
        print(f"Columns : {df.shape[1]}")


if __name__ == "__main__":
    data = load_raw_datasets()
    dataset_summary(data)
