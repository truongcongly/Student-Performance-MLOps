"""Reject malformed training data before it enters the model pipeline."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data.load import DATA_PATH, load_data


class DataValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = pd.read_csv(DATA_PATH, sep=";").head(1)

    def assert_invalid(self, data: pd.DataFrame) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "student-mat.csv"
            data.to_csv(path, sep=";", index=False)
            with self.assertRaises(ValueError):
                load_data(path)

    def test_missing_required_column(self) -> None:
        self.assert_invalid(self.row.drop(columns="studytime"))

    def test_missing_feature_value(self) -> None:
        data = self.row.copy()
        data.loc[data.index[0], "studytime"] = None
        self.assert_invalid(data)

    def test_grade_outside_target_range(self) -> None:
        data = self.row.copy()
        data.loc[data.index[0], "G3"] = 21
        self.assert_invalid(data)

    def test_unknown_yes_no_value(self) -> None:
        data = self.row.copy()
        data.loc[data.index[0], "schoolsup"] = "maybe"
        self.assert_invalid(data)


if __name__ == "__main__":
    unittest.main()
