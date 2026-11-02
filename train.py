"""Train Logistic Regression and Random Forest churn classifiers."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_preprocessing import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    load_or_create_clean_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
MODELS_PATH = MODEL_DIR / "churn_models.joblib"
EVALUATION_DATA_PATH = MODEL_DIR / "evaluation_data.joblib"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"


def build_preprocessor() -> ColumnTransformer:
    """Impute missing values, scale numeric columns, and encode categories."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def main() -> None:
    """Train both models and save fitted models plus a fixed evaluation split."""
    data = load_or_create_clean_dataset()
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    trained_models: dict[str, Pipeline] = {}
    for model_name, classifier in classifiers.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline
        print(f"Trained {model_name}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models, MODELS_PATH)
    joblib.dump(
        {"features": X_test, "target": y_test},
        EVALUATION_DATA_PATH,
    )

    # This provides a working app artifact until evaluate.py selects the F1 winner.
    joblib.dump(
        {
            "model": trained_models["Logistic Regression"],
            "model_name": "Logistic Regression",
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )
    print(f"Saved model candidates to {MODELS_PATH}")
    print(f"Saved holdout data to {EVALUATION_DATA_PATH}")
    print(f"Initial app model saved to {MODEL_PATH}; run evaluate.py to select the F1 winner.")


if __name__ == "__main__":
    main()
