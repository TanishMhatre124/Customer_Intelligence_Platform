"""
Build customer-level features and RFM segments from the cleaned datasets.

Mirrors the logic in notebooks/04_Feature_Engineering.ipynb and
notebooks/06_Customer_Segmentation.ipynb.
"""

import os

import pandas as pd

from src.config import (
    CHURN_FEATURES,
    ORDER_DATE_COLUMNS,
    PROCESSED_DATA_PATH,
    processed_path,
)


# -------------------------------
# Order-level features (Phase 4)
# -------------------------------

def add_order_level_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Add delivery_days, approval_time_hours, late_delivery, and
    delivery_difference_days to the orders DataFrame."""
    orders = orders.copy()

    for col in ORDER_DATE_COLUMNS:
        if orders[col].dtype == object:
            orders[col] = pd.to_datetime(orders[col])

    orders["delivery_days"] = (
        orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
    ).dt.days

    orders["approval_time_hours"] = (
        orders["order_approved_at"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600

    orders["late_delivery"] = (
        orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]
    )

    orders["delivery_difference_days"] = (
        orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
    ).dt.days

    return orders


def build_customer_features(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
) -> tuple:
    """
    Build one row per customer with total_orders, total_spent,
    average_order_value, and recency_days.

    Returns
    -------
    (customer_features, order_value, orders_with_features)
    """
    orders = add_order_level_features(orders)

    # Map orders to the stable customer_unique_id
    orders_customer = orders.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )

    # Frequency: orders per customer
    customer_frequency = (
        orders_customer.groupby("customer_unique_id")
        .size()
        .reset_index(name="total_orders")
    )

    # Monetary: total spend per order, then per customer
    order_items = order_items.copy()
    order_items["item_value"] = order_items["price"] + order_items["freight_value"]

    order_value = (
        order_items.groupby("order_id")["item_value"].sum().reset_index(name="order_value")
    )

    orders_customer = orders_customer.merge(order_value, on="order_id", how="left")

    customer_monetary = (
        orders_customer.groupby("customer_unique_id")["order_value"]
        .sum()
        .reset_index(name="total_spent")
    )

    customer_features = customer_frequency.merge(
        customer_monetary, on="customer_unique_id", how="left"
    )
    customer_features["average_order_value"] = (
        customer_features["total_spent"] / customer_features["total_orders"]
    )

    # Recency: days since last purchase, relative to the most recent order
    # date in the whole dataset (a fixed reference point for consistency).
    customer_recency = (
        orders_customer.groupby("customer_unique_id")["order_purchase_timestamp"]
        .max()
        .reset_index(name="last_purchase_date")
    )
    reference_date = orders["order_purchase_timestamp"].max()
    customer_recency["recency_days"] = (
        reference_date - customer_recency["last_purchase_date"]
    ).dt.days

    customer_features = customer_features.merge(
        customer_recency[["customer_unique_id", "recency_days"]],
        on="customer_unique_id",
        how="left",
    )

    return customer_features, order_value, orders_customer


def save_customer_features(customer_features: pd.DataFrame, order_value: pd.DataFrame,
                            orders_with_features: pd.DataFrame) -> None:
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    customer_features.to_csv(processed_path("customer_features"), index=False)
    order_value.to_csv(processed_path("order_value"), index=False)
    orders_with_features.to_csv(processed_path("orders_features"), index=False)
    print("Saved customer_features.csv, order_value.csv, orders_features.csv")


# -------------------------------
# RFM segmentation (Phase 6)
# -------------------------------

def build_rfm_segments(customer_features: pd.DataFrame) -> pd.DataFrame:
    """Compute RFM scores and assign a business-friendly customer segment."""
    rfm = customer_features[
        ["customer_unique_id", "recency_days", "total_orders", "total_spent"]
    ].copy()

    rfm.rename(
        columns={
            "recency_days": "Recency",
            "total_orders": "Frequency",
            "total_spent": "Monetary",
        },
        inplace=True,
    )

    # Lower recency = higher score (recent buyers are best)
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1])
    # Higher frequency = higher score
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
    # Higher spend = higher score
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5])

    rfm["R_Score"] = rfm["R_Score"].astype(str)
    rfm["F_Score"] = rfm["F_Score"].astype(str)
    rfm["M_Score"] = rfm["M_Score"].astype(str)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    rfm["Customer_Segment"] = rfm.apply(segment_customer, axis=1)

    return rfm


def segment_customer(row: pd.Series) -> str:
    """Assign a customer segment label based on R/F scores."""
    if row["R_Score"] == "5" and row["F_Score"] == "5":
        return "Champions"
    elif row["R_Score"] >= "4" and row["F_Score"] >= "4":
        return "Loyal Customers"
    elif row["R_Score"] >= "4":
        return "Potential Loyalists"
    elif row["R_Score"] == "3":
        return "Need Attention"
    elif row["R_Score"] <= "2" and row["F_Score"] >= "3":
        return "At Risk"
    else:
        return "Lost Customers"


def save_rfm_segments(rfm: pd.DataFrame) -> None:
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    rfm.to_csv(processed_path("rfm_customer_segments"), index=False)
    print("Saved rfm_customer_segments.csv")


if __name__ == "__main__":
    customers = pd.read_csv(processed_path("customers_clean"))
    orders = pd.read_csv(processed_path("orders_clean"))
    order_items = pd.read_csv(processed_path("order_items_clean"))

    customer_features, order_value, orders_features = build_customer_features(
        customers, orders, order_items
    )
    save_customer_features(customer_features, order_value, orders_features)

    rfm = build_rfm_segments(customer_features)
    save_rfm_segments(rfm)
