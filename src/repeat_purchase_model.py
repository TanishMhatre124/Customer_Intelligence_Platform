"""
Repeat-Purchase Prediction Model (leak-free churn model)
----------------------------------------------------------
The original churn model (src/model.py) defines churn from Recency and
Frequency and then predicts churn using those same features - that's
data leakage, explained in notebooks/07_Churn_Prediction.ipynb.

This module builds a genuinely predictive alternative:

- Uses only information known from a customer's FIRST order (order value,
  delivery experience, review score, payment method) - never anything
  computed from their full purchase history.
- Predicts whether they placed a SECOND order within 180 days of that
  first order - a real, forward-looking outcome, not a rule built from
  the same inputs.
- Customers whose first order is too recent for the 180-day outcome
  window to have closed are excluded (censored), rather than guessed at.

Expect meaningfully lower (and more honest) accuracy than the rule-based
model - predicting a stranger's future behavior from one order is
genuinely hard, and an AUC modestly above 0.5 here is a real result.
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.config import PROCESSED_DATA_PATH, RANDOM_STATE, TEST_SIZE, processed_path

REPEAT_PURCHASE_WINDOW_DAYS = 180

FIRST_ORDER_FEATURES = [
    "first_order_value",
    "first_delivery_days",
    "first_late_delivery",
    "first_approval_time_hours",
    "first_review_score",
    "first_payment_installments",
    "first_order_month",
    "first_order_dow",
]


def build_first_order_dataset(
    orders_features: pd.DataFrame,
    payments: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per customer: their first order's characteristics, plus
    whether they returned to buy again within REPEAT_PURCHASE_WINDOW_DAYS.
    """
    orders = orders_features.copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

    # Order each customer's orders chronologically
    orders = orders.sort_values(["customer_unique_id", "order_purchase_timestamp"])
    orders["order_rank"] = orders.groupby("customer_unique_id").cumcount() + 1

    first_orders = orders[orders["order_rank"] == 1].copy()
    second_orders = (
        orders[orders["order_rank"] == 2][["customer_unique_id", "order_purchase_timestamp"]]
        .rename(columns={"order_purchase_timestamp": "second_purchase_date"})
    )

    # Payment info per order (aggregate multiple installments rows)
    payment_agg = (
        payments.groupby("order_id")
        .agg(payment_installments=("payment_installments", "max"))
        .reset_index()
    )

    # Review score per order (an order can have >1 review; take the first)
    review_agg = reviews.groupby("order_id")["review_score"].first().reset_index()

    data = first_orders.merge(payment_agg, on="order_id", how="left")
    data = data.merge(review_agg, on="order_id", how="left")
    data = data.merge(second_orders, on="customer_unique_id", how="left")

    # Outcome: did they buy again within the window?
    data["days_to_second_purchase"] = (
        data["second_purchase_date"] - data["order_purchase_timestamp"]
    ).dt.days

    data["repeat_purchase_180d"] = np.where(
        data["days_to_second_purchase"].notna()
        & (data["days_to_second_purchase"] <= REPEAT_PURCHASE_WINDOW_DAYS),
        1,
        0,
    )

    # Censoring: only keep customers whose first order is old enough that
    # we'd have observed a repeat purchase by now if one happened.
    reference_date = orders["order_purchase_timestamp"].max()
    data["days_since_first_order"] = (reference_date - data["order_purchase_timestamp"]).dt.days
    data = data[data["days_since_first_order"] >= REPEAT_PURCHASE_WINDOW_DAYS].copy()

    # Feature engineering from the first order only
    data["first_order_value"] = data["order_value"]
    data["first_delivery_days"] = data["delivery_days"]
    data["first_late_delivery"] = data["late_delivery"].astype(float)
    data["first_approval_time_hours"] = data["approval_time_hours"]
    data["first_review_score"] = data["review_score"]
    data["first_payment_installments"] = data["payment_installments"]
    data["first_order_month"] = data["order_purchase_timestamp"].dt.month
    data["first_order_dow"] = data["order_purchase_timestamp"].dt.dayofweek

    # Impute missing values with sensible defaults (median for numeric)
    for col in FIRST_ORDER_FEATURES:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())

    keep_cols = ["customer_unique_id", "order_id"] + FIRST_ORDER_FEATURES + ["repeat_purchase_180d"]
    return data[keep_cols].reset_index(drop=True)


def split_features(data: pd.DataFrame):
    X = data[FIRST_ORDER_FEATURES]
    y = data["repeat_purchase_180d"]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def get_candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5, min_samples_leaf=50),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_estimators=300, max_depth=8, min_samples_leaf=30
        ),
    }


def train_models(X_train, y_train, models: dict = None) -> dict:
    models = models or get_candidate_models()
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def save_outputs(data: pd.DataFrame, model, model_name: str) -> None:
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    X = data[FIRST_ORDER_FEATURES]
    data = data.copy()
    data["Repeat_Purchase_Probability"] = model.predict_proba(X)[:, 1]
    data["Predicted_Repeat_Purchase"] = (data["Repeat_Purchase_Probability"] >= 0.5).astype(int)
    data.to_csv(os.path.join(PROCESSED_DATA_PATH, "first_order_features_scored.csv"), index=False)
    print(f"Saved first_order_features_scored.csv using {model_name}")
