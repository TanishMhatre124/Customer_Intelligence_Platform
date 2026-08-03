"""
Genuine, leakage-free repeat-purchase / retention model.

The Phase 7 churn model (see model.py) defines churn from Recency/Frequency and
then predicts it from Recency/Frequency/Monetary -- which is why it scores
near-perfectly but isn't a real forecast (see the caveat in
notebooks/07_Churn_Prediction.ipynb).

This module instead:
- Uses only a customer's FIRST order to build features (nothing that requires
  hindsight about their whole history).
- Defines the label as a genuinely future event: did they place a SECOND order
  within 90 days of their first one?
- Only scores customers whose first order happened at least 90 days before the
  dataset's cutoff date, so every customer had a fair chance to reorder.

This is a much harder, much more honest problem: only ~2% of eligible
customers actually reorder within 90 days, so the model is evaluated on
precision/recall/ROC-AUC rather than accuracy (a model that always predicts
"no repeat" would already be ~98% "accurate" and useless).
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.config import PROCESSED_DATA_PATH, RANDOM_STATE, TEST_SIZE, processed_path

REPEAT_WINDOW_DAYS = 90
TOP_N_CATEGORIES = 10


# -------------------------------------------------------------------
# Label: did the customer place a second order within REPEAT_WINDOW_DAYS?
# -------------------------------------------------------------------

def build_repeat_purchase_label(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per eligible customer:
    customer_unique_id, first_order_id, first_order_date, repeat_within_90
    """
    orders = orders.copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders_sorted = orders.sort_values(["customer_unique_id", "order_purchase_timestamp"])
    orders_sorted["order_rank"] = orders_sorted.groupby("customer_unique_id").cumcount() + 1

    first = orders_sorted[orders_sorted["order_rank"] == 1][
        ["customer_unique_id", "order_id", "order_purchase_timestamp"]
    ].rename(columns={"order_id": "first_order_id", "order_purchase_timestamp": "first_order_date"})

    second = orders_sorted[orders_sorted["order_rank"] == 2][
        ["customer_unique_id", "order_purchase_timestamp"]
    ].rename(columns={"order_purchase_timestamp": "second_order_date"})

    merged = first.merge(second, on="customer_unique_id", how="left")
    merged["days_to_second_order"] = (
        merged["second_order_date"] - merged["first_order_date"]
    ).dt.days
    merged["repeat_within_90"] = (merged["days_to_second_order"] <= REPEAT_WINDOW_DAYS).fillna(False).astype(int)

    reference_date = orders["order_purchase_timestamp"].max()
    cutoff = reference_date - pd.Timedelta(days=REPEAT_WINDOW_DAYS)
    eligible = merged[merged["first_order_date"] <= cutoff].copy()

    return eligible[["customer_unique_id", "first_order_id", "first_order_date", "repeat_within_90"]]


# -------------------------------------------------------------------
# Features: everything known from the customer's FIRST order only
# -------------------------------------------------------------------

def build_first_order_features(
    labels: pd.DataFrame,
    orders: pd.DataFrame,
    reviews: pd.DataFrame,
    payments: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    df = labels.merge(
        orders[
            ["order_id", "delivery_days", "late_delivery", "delivery_difference_days",
             "approval_time_hours", "order_value"]
        ],
        left_on="first_order_id", right_on="order_id", how="left",
    )

    # Review score for the first order (if reviewed yet)
    review_score = reviews.groupby("order_id")["review_score"].mean().reset_index()
    df = df.merge(review_score, left_on="first_order_id", right_on="order_id", how="left", suffixes=("", "_rev"))
    df["review_score"] = df["review_score"].fillna(df["review_score"].median())

    # Payment behavior (first payment record per order)
    pay = (
        payments.sort_values("payment_sequential")
        .groupby("order_id")
        .first()[["payment_type", "payment_installments"]]
        .reset_index()
    )
    df = df.merge(pay, left_on="first_order_id", right_on="order_id", how="left", suffixes=("", "_pay"))

    # Product category (most common item category on the first order) + price/freight
    items = order_items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    item_agg = items.groupby("order_id").agg(
        avg_price=("price", "mean"),
        avg_freight=("freight_value", "mean"),
        n_items=("product_id", "count"),
    ).reset_index()
    top_category = (
        items.groupby("order_id")["product_category_name"]
        .agg(lambda x: x.mode().iat[0] if not x.mode().empty else "unknown")
        .reset_index()
        .rename(columns={"product_category_name": "category"})
    )
    df = df.merge(item_agg, left_on="first_order_id", right_on="order_id", how="left", suffixes=("", "_items"))
    df = df.merge(top_category, left_on="first_order_id", right_on="order_id", how="left", suffixes=("", "_cat"))

    # Collapse long-tail categories to keep the feature space manageable
    top_categories = df["category"].value_counts().head(TOP_N_CATEGORIES).index
    df["category"] = df["category"].where(df["category"].isin(top_categories), "other")

    # Seasonality
    df["purchase_month"] = df["first_order_date"].dt.month
    df["purchase_dayofweek"] = df["first_order_date"].dt.dayofweek

    # Drop join helper columns
    drop_cols = [c for c in df.columns if c.startswith("order_id")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Fill remaining gaps
    numeric_fill = {
        "delivery_days": df["delivery_days"].median(),
        "delivery_difference_days": df["delivery_difference_days"].median(),
        "approval_time_hours": df["approval_time_hours"].median(),
        "avg_price": df["avg_price"].median(),
        "avg_freight": df["avg_freight"].median(),
        "payment_installments": 1,
    }
    df = df.fillna(numeric_fill)
    df["payment_type"] = df["payment_type"].fillna("unknown")
    df["late_delivery"] = df["late_delivery"].astype(str).map({"True": 1, "False": 0}).fillna(0)

    return df


NUMERIC_FEATURES = [
    "delivery_days", "late_delivery", "delivery_difference_days", "approval_time_hours",
    "order_value", "review_score", "payment_installments", "avg_price", "avg_freight",
    "n_items", "purchase_month", "purchase_dayofweek",
]
CATEGORICAL_FEATURES = ["payment_type", "category"]


def prepare_model_matrix(df: pd.DataFrame):
    """One-hot encode categorical features, return X, y, and feature names."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_encoded = encoder.fit_transform(df[CATEGORICAL_FEATURES])
    cat_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)

    X = np.hstack([df[NUMERIC_FEATURES].values, cat_encoded])
    feature_names = NUMERIC_FEATURES + list(cat_names)
    y = df["repeat_within_90"].values

    return X, y, feature_names, encoder


def train_retention_model(df: pd.DataFrame):
    X, y, feature_names, encoder = prepare_model_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, feature_names, encoder


if __name__ == "__main__":
    orders = pd.read_csv(processed_path("orders_features"))
    reviews = pd.read_csv(processed_path("reviews_clean"))
    payments = pd.read_csv(processed_path("payments_clean"))
    order_items = pd.read_csv(processed_path("order_items_clean"))
    products = pd.read_csv(processed_path("products_clean"))

    labels = build_repeat_purchase_label(orders)
    print(f"Eligible customers: {len(labels):,}")
    print(f"Repeat-within-90-days rate: {labels['repeat_within_90'].mean() * 100:.2f}%")

    features = build_first_order_features(labels, orders, reviews, payments, order_items, products)
    model, X_train, X_test, y_train, y_test, feature_names, encoder = train_retention_model(features)

    from sklearn.metrics import classification_report, roc_auc_score

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    print(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print(classification_report(y_test, y_pred, target_names=["No Repeat", "Repeat"]))

    importance = pd.DataFrame({
        "feature": feature_names, "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    features.to_csv(processed_path("retention_features"), index=False)

    # Score the full eligible population (not just the test split) for the dashboard
    X_full, y_full, _, _ = prepare_model_matrix(features)
    features["Repeat_Probability"] = model.predict_proba(X_full)[:, 1]
    features.to_csv(processed_path("retention_scored"), index=False)
    print("\nSaved retention_model_features.csv and retention_model_scored.csv")
