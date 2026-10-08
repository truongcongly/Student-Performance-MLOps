"""Checks for the feature boundary and train-only preprocessing."""

import unittest

from sklearn.model_selection import train_test_split

from src.data.load import FEATURES, load_data
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


if __name__ == "__main__":
    unittest.main()
