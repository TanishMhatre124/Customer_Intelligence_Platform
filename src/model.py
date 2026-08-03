"""
Churn prediction model: label creation, train/test split, and training of
Logistic Regression, Decision Tree, and Random Forest classifiers.

Mirrors and completes notebooks/07_Churn_Prediction.ipynb.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CHURN_FEATURES,
    CHURN_FREQUENCY_THRESHOLD,
    CHURN_RECENCY_THRESHOLD_DAYS,
    MODELS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)


def add_churn_label(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Define churn as: no purchase in the last CHURN_RECENCY_THRESHOLD_DAYS
    days AND at most CHURN_FREQUENCY_THRESHOLD lifetime orders.

    This captures customers who bought rarely and have gone quiet, as
    opposed to loyal customers who simply haven't needed to reorder yet.
    """
    rfm = rfm.copy()
    rfm["Churn"] = np.where(
        (rfm["Recency"] > CHURN_RECENCY_THRESHOLD_DAYS)
        & (rfm["Frequency"] <= CHURN_FREQUENCY_THRESHOLD),
        1,
        0,
    )
    return rfm


def split_features(rfm: pd.DataFrame):
    """Split into train/test sets using the RFM features to predict Churn."""
    X = rfm[CHURN_FEATURES]
    y = rfm["Churn"]

    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def get_candidate_models() -> dict:
    """Return the three candidate classifiers used in this project."""
    return {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_estimators=200, max_depth=8
        ),
    }


def train_models(X_train, y_train, models: dict = None) -> dict:
    """Fit each candidate model on the training data."""
    models = models or get_candidate_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def save_model(model, filename: str = "churn_model_random_forest.joblib") -> str:
    os.makedirs(MODELS_PATH, exist_ok=True)
    path = os.path.join(MODELS_PATH, filename)
    joblib.dump(model, path)
    print(f"Saved model to {path}")
    return path


def predict_high_risk_customers(rfm: pd.DataFrame, model, threshold: float = 0.5) -> pd.DataFrame:
    """
    Score every customer with a fitted model and return those predicted
    as high churn risk, sorted by predicted probability descending.
    """
    X = rfm[CHURN_FEATURES]
    probabilities = model.predict_proba(X)[:, 1]

    scored = rfm.copy()
    scored["Churn_Probability"] = probabilities
    scored["Predicted_Churn"] = (probabilities >= threshold).astype(int)

    high_risk = scored[scored["Predicted_Churn"] == 1].sort_values(
        "Churn_Probability", ascending=False
    )
    return high_risk
