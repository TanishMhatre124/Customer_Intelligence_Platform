"""
Customer Intelligence & Churn Prediction Platform
--------------------------------------------------
End-to-end pipeline: load raw data -> clean -> engineer features ->
segment customers (RFM) -> train & evaluate churn models -> save outputs.

Run from the project root:
    python main.py
"""

import os

import pandas as pd

from src import config
from src.data_cleaning import clean_datasets, save_cleaned_datasets
from src.data_loader import dataset_summary, load_raw_datasets
from src.evaluation import compare_models, feature_importance
from src.feature_engineering import (
    build_customer_features,
    build_rfm_segments,
    save_customer_features,
    save_rfm_segments,
)
from src.model import (
    add_churn_label,
    get_candidate_models,
    predict_high_risk_customers,
    save_model,
    split_features,
    train_models,
)
from src.retention_model import (
    build_first_order_features,
    build_repeat_purchase_label,
    prepare_model_matrix,
    train_retention_model,
)
from src.utils import print_section


def run_pipeline():
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(config.MODELS_PATH, exist_ok=True)
    os.makedirs(config.REPORTS_PATH, exist_ok=True)

    # 1. Load raw data
    print_section("Step 1: Loading raw data")
    raw = load_raw_datasets()
    dataset_summary(raw)

    # 2. Clean
    print_section("Step 2: Cleaning data")
    cleaned = clean_datasets(raw)
    save_cleaned_datasets(cleaned)

    # 3. Feature engineering
    print_section("Step 3: Building customer features")
    customer_features, order_value, orders_features = build_customer_features(
        cleaned["customers"], cleaned["orders"], cleaned["order_items"]
    )
    save_customer_features(customer_features, order_value, orders_features)

    # 4. RFM segmentation
    print_section("Step 4: RFM customer segmentation")
    rfm = build_rfm_segments(customer_features)
    save_rfm_segments(rfm)
    print(rfm["Customer_Segment"].value_counts())

    # 5. Churn modeling
    print_section("Step 5: Training churn models")
    rfm_labeled = add_churn_label(rfm)
    print(f"Churn rate: {rfm_labeled['Churn'].mean() * 100:.2f}%")

    X_train, X_test, y_train, y_test = split_features(rfm_labeled)
    models = train_models(X_train, y_train, get_candidate_models())

    results = compare_models(models, X_test, y_test)
    print("\nModel comparison (sorted by ROC-AUC):")
    print(results.to_string(index=False))
    results.to_csv(os.path.join(config.REPORTS_PATH, "model_comparison.csv"), index=False)

    best_name = results.iloc[0]["model"]
    best_model = models[best_name]
    print(f"\nBest model: {best_name}")

    importance = feature_importance(best_model, config.CHURN_FEATURES)
    print("\nFeature importance:")
    print(importance.to_string(index=False))

    safe_name = best_name.lower().replace(" ", "_")
    save_model(best_model, filename=f"churn_model_{safe_name}.joblib")

    # 6. High-risk customers
    print_section("Step 6: Identifying high-risk customers")
    high_risk = predict_high_risk_customers(rfm_labeled, best_model)
    high_risk.to_csv(os.path.join(config.REPORTS_PATH, "high_risk_customers.csv"), index=False)
    print(f"Flagged {len(high_risk)} customers as high churn risk.")
    print(f"Saved to reports/high_risk_customers.csv")

    # Save the full scored RFM table (used by the dashboard / Streamlit app)
    rfm_labeled["Churn_Probability"] = best_model.predict_proba(rfm_labeled[config.CHURN_FEATURES])[:, 1]
    rfm_labeled["Predicted_Churn"] = (rfm_labeled["Churn_Probability"] >= 0.5).astype(int)
    scored_path = os.path.join(config.PROCESSED_DATA_PATH, "rfm_customer_segments_scored.csv")
    rfm_labeled.to_csv(scored_path, index=False)
    print(f"Saved rfm_customer_segments_scored.csv")

    # 7. Leakage-free retention model (real future outcome, first-order-only features)
    print_section("Step 7: Training leakage-free retention model")
    orders_raw = pd.read_csv(config.processed_path("orders_features"))
    reviews = pd.read_csv(config.processed_path("reviews_clean"))
    payments = pd.read_csv(config.processed_path("payments_clean"))
    order_items = pd.read_csv(config.processed_path("order_items_clean"))
    products = pd.read_csv(config.processed_path("products_clean"))

    repeat_labels = build_repeat_purchase_label(orders_raw)
    print(f"Eligible first-time customers: {len(repeat_labels):,}")
    print(f"Repeat-within-90-days rate: {repeat_labels['repeat_within_90'].mean() * 100:.2f}%")

    retention_features = build_first_order_features(
        repeat_labels, orders_raw, reviews, payments, order_items, products
    )
    (retention_model, _, X_test_r, _, y_test_r, feature_names_r, _) = train_retention_model(
        retention_features
    )

    from sklearn.metrics import roc_auc_score
    y_proba_r = retention_model.predict_proba(X_test_r)[:, 1]
    print(f"Retention model ROC-AUC: {roc_auc_score(y_test_r, y_proba_r):.3f}")

    X_full, _, _, _ = prepare_model_matrix(retention_features)
    retention_features["Repeat_Probability"] = retention_model.predict_proba(X_full)[:, 1]
    retention_features.to_csv(config.processed_path("retention_features"), index=False)
    retention_features.to_csv(config.processed_path("retention_scored"), index=False)
    print("Saved retention_model_features.csv and retention_model_scored.csv")

    print_section("Pipeline complete")


if __name__ == "__main__":
    run_pipeline()
