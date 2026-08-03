"""
Project configuration: paths and constants shared across the pipeline.
"""

import os

# -------------------------------
# Project Paths
# -------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
REPORTS_PATH = os.path.join(PROJECT_ROOT, "reports")

# -------------------------------
# Raw Dataset Files
# -------------------------------

RAW_FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# -------------------------------
# Processed Dataset Files
# -------------------------------

PROCESSED_FILES = {
    "customers_clean": "customers_clean.csv",
    "orders_clean": "orders_clean.csv",
    "order_items_clean": "order_items_clean.csv",
    "payments_clean": "payments_clean.csv",
    "reviews_clean": "reviews_clean.csv",
    "products_clean": "products_clean.csv",
    "sellers_clean": "sellers_clean.csv",
    "customer_features": "customer_features.csv",
    "order_value": "order_value.csv",
    "orders_features": "orders_features.csv",
    "rfm_customer_segments": "rfm_customer_segments.csv",
    "retention_features": "retention_model_features.csv",
    "retention_scored": "retention_model_scored.csv",
}

# -------------------------------
# Date Columns
# -------------------------------

ORDER_DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

REVIEW_DATE_COLUMNS = [
    "review_creation_date",
    "review_answer_timestamp",
]

# -------------------------------
# Churn Definition
# -------------------------------

CHURN_RECENCY_THRESHOLD_DAYS = 180
CHURN_FREQUENCY_THRESHOLD = 2

# -------------------------------
# Modeling
# -------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
CHURN_FEATURES = ["Recency", "Frequency", "Monetary"]


def raw_path(name: str) -> str:
    """Return the full path to a raw dataset file."""
    return os.path.join(RAW_DATA_PATH, RAW_FILES[name])


def processed_path(name: str) -> str:
    """Return the full path to a processed dataset file."""
    return os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILES[name])
