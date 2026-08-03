"""
Clean raw Olist datasets: fix data types, inspect missing values / duplicates,
and save cleaned outputs to data/processed.

Mirrors the logic in notebooks/03_Data_Cleaning.ipynb.
"""

import os

import pandas as pd

from src.config import (
    ORDER_DATE_COLUMNS,
    PROCESSED_DATA_PATH,
    REVIEW_DATE_COLUMNS,
    processed_path,
)


def convert_date_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Convert the given columns of a DataFrame to datetime, in place."""
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])
    return df


def inspect_data_quality(datasets: dict, verbose: bool = True) -> dict:
    """
    Report shape, dtypes, missing values, and duplicate counts for each
    dataset. Returns a dict of {name: {"missing": Series, "duplicates": int}}.
    """
    report = {}

    for name, df in datasets.items():
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        duplicates = int(df.duplicated().sum())

        report[name] = {"missing": missing, "duplicates": duplicates}

        if verbose and (len(missing) > 0 or duplicates > 0):
            print("=" * 70)
            print(name.upper())
            print("=" * 70)
            if len(missing) > 0:
                print("Missing Values:")
                print(missing)
            print(f"Duplicate Rows: {duplicates}")

    return report


def clean_datasets(datasets: dict) -> dict:
    """
    Apply the cleaning steps used in the project:
    - Convert order/review date columns to datetime.
    - Missing values in products/reviews are kept (they carry business
      meaning - e.g. missing review text is optional feedback, and
      products with missing descriptors are still used in real orders).
    - No duplicate rows were found in the source data, so none are dropped
      here; the duplicate check is still run so regressions are caught.

    Returns the same dict with cleaned DataFrames.
    """
    if "orders" in datasets:
        convert_date_columns(datasets["orders"], ORDER_DATE_COLUMNS)

    if "reviews" in datasets:
        convert_date_columns(datasets["reviews"], REVIEW_DATE_COLUMNS)

    inspect_data_quality(datasets)

    return datasets


def save_cleaned_datasets(datasets: dict) -> None:
    """Save cleaned datasets to data/processed as `<name>_clean.csv`."""
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    name_map = {
        "customers": "customers_clean",
        "orders": "orders_clean",
        "order_items": "order_items_clean",
        "payments": "payments_clean",
        "reviews": "reviews_clean",
        "products": "products_clean",
        "sellers": "sellers_clean",
    }

    for raw_name, processed_name in name_map.items():
        if raw_name in datasets:
            datasets[raw_name].to_csv(processed_path(processed_name), index=False)
            print(f"Saved {processed_name}.csv")


if __name__ == "__main__":
    from src.data_loader import load_raw_datasets

    data = load_raw_datasets()
    data = clean_datasets(data)
    save_cleaned_datasets(data)
