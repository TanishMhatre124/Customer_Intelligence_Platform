"""
Small shared helpers used across the pipeline and notebooks.
"""

import pandas as pd


def value_counts_table(series: pd.Series, count_name: str = "Count") -> pd.DataFrame:
    """Return a tidy value_counts DataFrame with percentage column."""
    counts = series.value_counts().reset_index()
    counts.columns = [series.name or "value", count_name]
    counts["Percentage"] = (counts[count_name] / counts[count_name].sum() * 100).round(2)
    return counts


def print_section(title: str, width: int = 70) -> None:
    """Print a formatted section header, used for readable console output."""
    print("=" * width)
    print(title.upper())
    print("=" * width)


def missing_value_report(df: pd.DataFrame) -> pd.Series:
    """Return only the columns of `df` that have missing values, with counts."""
    missing = df.isnull().sum()
    return missing[missing > 0]
