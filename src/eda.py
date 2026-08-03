"""
Reusable EDA / summary helpers shared between notebooks/05_Exploratory_Data_Analysis.ipynb
and the dashboard notebook.
"""

import pandas as pd


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Customer count, % of base, and total revenue per RFM segment."""
    summary = (
        rfm.groupby("Customer_Segment")
        .agg(
            Customer_Count=("customer_unique_id", "count"),
            Total_Revenue=("Monetary", "sum"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
        )
        .reset_index()
    )
    summary["Pct_of_Customers"] = (
        summary["Customer_Count"] / summary["Customer_Count"].sum() * 100
    ).round(2)
    summary["Pct_of_Revenue"] = (
        summary["Total_Revenue"] / summary["Total_Revenue"].sum() * 100
    ).round(2)
    return summary.sort_values("Total_Revenue", ascending=False).reset_index(drop=True)


def monthly_revenue(orders_features: pd.DataFrame) -> pd.DataFrame:
    """Total order_value by calendar month, for trend charts."""
    df = orders_features.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    return (
        df.groupby("month")["order_value"]
        .sum()
        .reset_index()
        .sort_values("month")
    )
