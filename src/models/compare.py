"""Select a model by training folds, then evaluate once on held-out data."""

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from src.data.load import load_data
from src.models.train import build_pipeline


SEED = 42


def main() -> None:
    features, target = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=SEED
    )
    candidates = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=1
        ),
    }
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_mae = {}
    print("5-fold cross-validation MAE on training data (lower is better):")
    for name, estimator in candidates.items():
        scores = -cross_val_score(
            build_pipeline(estimator), x_train, y_train,
            cv=folds, scoring="neg_mean_absolute_error", n_jobs=1,
        )
        cv_mae[name] = float(np.mean(scores))
        print(f"  {name}: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

    best_name = min(cv_mae, key=cv_mae.get)
    model = build_pipeline(candidates[best_name]).fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(f"Selected by CV: {best_name}")
    print(f"Held-out test MAE: {mean_absolute_error(y_test, predictions):.3f}")
    print(f"Held-out test RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.3f}")
    print(f"Held-out test R2: {r2_score(y_test, predictions):.3f}")


if __name__ == "__main__":
    main()
