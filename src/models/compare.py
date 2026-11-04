"""Select a model by training folds, then evaluate once on held-out data."""

import argparse
import os
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from src.data.load import DATA_PATH, FEATURES, load_data
from src.models.train import build_pipeline


SEED = 42
MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "student_grade_pipeline.joblib"
EXPERIMENT_NAME = "student-performance-early-prediction"


def save_and_verify(model, sample, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = model.predict(sample)
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(restored.predict(sample), expected)


def main(track: bool = False) -> None:
    mlflow = None
    if track:
        import mlflow as tracking

        mlflow = tracking
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
        mlflow.set_experiment(EXPERIMENT_NAME)
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
    trained_models = {}
    run_ids = {}
    data_hash = sha256(DATA_PATH.read_bytes()).hexdigest()
    print("5-fold cross-validation MAE on training data (lower is better):")
    for name, estimator in candidates.items():
        scores = -cross_val_score(
            build_pipeline(estimator), x_train, y_train,
            cv=folds, scoring="neg_mean_absolute_error", n_jobs=1,
        )
        cv_mae[name] = float(np.mean(scores))
        print(f"  {name}: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

        model = build_pipeline(estimator).fit(x_train, y_train)
        if mlflow is not None:
            with mlflow.start_run(run_name=name.lower().replace(" ", "-")) as run:
                run_ids[name] = run.info.run_id
                mlflow.set_tags({"dataset_sha256": data_hash, "prediction_time": "course-start"})
                mlflow.log_params({
                    "model_type": type(estimator).__name__,
                    "random_state_split": SEED,
                    "test_size": 0.2,
                    "cv_folds": 5,
                    "feature_count": len(FEATURES),
                    **{f"model_{key}": str(value) for key, value in estimator.get_params(deep=False).items()},
                })
                mlflow.log_metrics({"cv_mae": cv_mae[name], "cv_mae_std": float(np.std(scores))})
                with TemporaryDirectory() as directory:
                    artifact = Path(directory) / "student_grade_pipeline.joblib"
                    joblib.dump(model, artifact)
                    mlflow.log_artifact(str(artifact), artifact_path="model")
        trained_models[name] = model

    best_name = min(cv_mae, key=cv_mae.get)
    model = trained_models[best_name]
    predictions = model.predict(x_test)
    test_metrics = {
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "test_r2": float(r2_score(y_test, predictions)),
    }
    if mlflow is not None:
        with mlflow.start_run(run_id=run_ids[best_name]):
            mlflow.set_tag("selected_by_cv", "true")
            mlflow.log_metrics(test_metrics)
    print(f"Selected by CV: {best_name}")
    if mlflow is not None:
        print(f"Selected MLflow run ID: {run_ids[best_name]}")
    print(f"Held-out test MAE: {test_metrics['test_mae']:.3f}")
    print(f"Held-out test RMSE: {test_metrics['test_rmse']:.3f}")
    print(f"Held-out test R2: {test_metrics['test_r2']:.3f}")
    save_and_verify(model, x_test.head(5))
    print(f"Saved and verified pipeline: {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", action="store_true", help="Log runs to the MLflow server")
    main(parser.parse_args().track)
