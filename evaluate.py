"""Evaluate trained churn models and select the best by churn-class F1-score."""

import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data_preprocessing import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
MODELS_PATH = MODEL_DIR / "churn_models.joblib"
EVALUATION_DATA_PATH = MODEL_DIR / "evaluation_data.joblib"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"


def calculate_metrics(model, features, target) -> dict[str, object]:
    """Calculate classification metrics with churn ("Yes") as the positive class."""
    predictions = model.predict(features)
    matrix = confusion_matrix(target, predictions, labels=["No", "Yes"])
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(
            precision_score(target, predictions, pos_label="Yes", zero_division=0)
        ),
        "recall": float(
            recall_score(target, predictions, pos_label="Yes", zero_division=0)
        ),
        "f1_score": float(
            f1_score(target, predictions, pos_label="Yes", zero_division=0)
        ),
        "confusion_matrix": {
            "labels": ["No", "Yes"],
            "rows": matrix.tolist(),
        },
    }


def main() -> None:
    """Evaluate every saved model, print results, and persist the F1-score winner."""
    if not MODELS_PATH.exists() or not EVALUATION_DATA_PATH.exists():
        raise FileNotFoundError("Run `python train.py` before evaluating the models.")

    models = joblib.load(MODELS_PATH)
    holdout = joblib.load(EVALUATION_DATA_PATH)
    features = holdout["features"]
    target = holdout["target"]

    results = {
        name: calculate_metrics(model, features, target)
        for name, model in models.items()
    }
    for name, metrics in results.items():
        print(f"\n{name}")
        for metric in ("accuracy", "precision", "recall", "f1_score"):
            print(f"  {metric.replace('_', ' ').title()}: {metrics[metric]:.3f}")
        print("  Confusion matrix (rows=true [No, Yes], columns=predicted [No, Yes]):")
        print(metrics["confusion_matrix"]["rows"])

    best_name = max(results, key=lambda name: results[name]["f1_score"])
    joblib.dump(
        {
            "model": models[best_name],
            "model_name": best_name,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": results[best_name],
        },
        MODEL_PATH,
    )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(results, metrics_file, indent=2)

    print(f"\nSelected {best_name} by churn F1-score and saved it to {MODEL_PATH}")
    print(f"Saved all evaluation results to {METRICS_PATH}")


if __name__ == "__main__":
    main()
