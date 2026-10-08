"""Train the first reproducible regression baseline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_data


def build_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORICAL_FEATURES)),
    ])
    return Pipeline([("preprocessing", preprocessing), ("model", Ridge(alpha=1.0))])


def main() -> None:
    features, target = load_data()
    x_train, x_test, y_train, _ = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    train_mae = mean_absolute_error(y_train, pipeline.predict(x_train))
    print(f"Train rows: {len(x_train)}; held-out test rows: {len(x_test)}")
    print(f"Ridge training MAE: {train_mae:.3f} grade points")
    print("Test set reserved for model comparison and evaluation in the next step.")


if __name__ == "__main__":
    main()
