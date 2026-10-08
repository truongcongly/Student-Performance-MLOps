"""Checks for the feature boundary and train-only preprocessing."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np

from sklearn.model_selection import train_test_split

from src.data.load import FEATURES, load_data
from src.models.compare import save_and_verify
from src.models.train import build_pipeline


class BaselineTests(unittest.TestCase):
    def test_features_exclude_future_information(self) -> None:
        self.assertTrue({"G1", "G2", "absences"}.isdisjoint(FEATURES))
        features, target = load_data()
        self.assertEqual(features.shape, (395, 12))
        self.assertEqual(len(target), 395)

    def test_pipeline_predicts_on_held_out_rows(self) -> None:
        features, target = load_data()
        x_train, x_test, y_train, _ = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        self.assertTrue(set(x_train.index).isdisjoint(x_test.index))
        model = build_pipeline().fit(x_train, y_train)
        self.assertEqual(len(model.predict(x_test)), 79)

    def test_saved_pipeline_preserves_predictions(self) -> None:
        features, target = load_data()
        model = build_pipeline().fit(features.iloc[:100], target.iloc[:100])
        sample = features.iloc[100:105]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.joblib"
            save_and_verify(model, sample, path)
            self.assertTrue(path.is_file())
            np.testing.assert_allclose(joblib.load(path).predict(sample), model.predict(sample))


if __name__ == "__main__":
    unittest.main()
