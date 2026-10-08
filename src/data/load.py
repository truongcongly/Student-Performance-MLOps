"""Load the mathematics data for the early prediction baseline."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "student-mat.csv"
NUMERIC_FEATURES = ("Medu", "Fedu", "traveltime", "studytime", "failures")
CATEGORICAL_FEATURES = (
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet"
)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "G3"


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path, sep=";")
    required = set(FEATURES) | {TARGET}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data.empty or data[list(FEATURES) + [TARGET]].isna().any().any():
        raise ValueError("Dataset is empty or has missing feature/target values")
    if not data[TARGET].between(0, 20).all():
        raise ValueError("G3 must be between 0 and 20")
    for column in CATEGORICAL_FEATURES:
        if not data[column].isin(("yes", "no")).all():
            raise ValueError(f"Unexpected values in {column}; expected yes/no")

    return data.loc[:, list(FEATURES)], data[TARGET]
