"""Download and clean the public Telco customer churn dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_customer_churn_clean.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/arghac14/Customer-Churn-Analysis/"
    "master/Decision%20Tree/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

TARGET_COLUMN = "Churn"
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in NUMERIC_COLUMNS
]


def load_raw_dataset() -> pd.DataFrame:
    """Read the cached source CSV or download it from the public dataset mirror."""
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists():
        return pd.read_csv(RAW_DATA_PATH)

    print(f"Downloading the public churn dataset from {DATA_URL}")
    data = pd.read_csv(DATA_URL)
    data.to_csv(RAW_DATA_PATH, index=False)
    return data


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, clean numeric values, and return model-ready data."""
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    cleaned = data[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned[NUMERIC_COLUMNS] = cleaned[NUMERIC_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(str).str.strip()
    cleaned = cleaned[cleaned[TARGET_COLUMN].isin(["Yes", "No"])]
    cleaned = cleaned.reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("No rows have a valid Yes/No churn target.")
    return cleaned


def load_or_create_clean_dataset() -> pd.DataFrame:
    """Load processed data when available, otherwise clean and save the raw data."""
    if CLEAN_DATA_PATH.exists():
        return pd.read_csv(CLEAN_DATA_PATH)

    cleaned = clean_dataset(load_raw_dataset())
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_DATA_PATH, index=False)
    return cleaned


def main() -> None:
    """Clean the source dataset and save a reusable processed CSV."""
    cleaned = clean_dataset(load_raw_dataset())
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Saved {len(cleaned):,} cleaned customer records to {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()
