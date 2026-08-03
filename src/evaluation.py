"""
Model evaluation helpers: metrics, comparison table, and feature importance.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test, name: str = "Model") -> dict:
    """Compute accuracy, precision, recall, F1, and ROC-AUC for a fitted model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    return metrics


def compare_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate every model in `models` and return a sorted comparison table."""
    results = [evaluate_model(model, X_test, y_test, name) for name, model in models.items()]
    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return results_df


def print_classification_report(model, X_test, y_test, name: str = "Model") -> None:
    y_pred = model.predict(X_test)
    print(f"\n{name} - Classification Report")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Return a DataFrame of feature importances (tree models) or absolute
    coefficients (linear models), sorted descending.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])
    else:
        raise ValueError("Model does not expose feature_importances_ or coef_")

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
