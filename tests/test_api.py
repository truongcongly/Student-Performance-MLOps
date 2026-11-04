"""HTTP contract checks for the saved prediction pipeline."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app
from src.data.load import FEATURES


EXAMPLE = Path(__file__).resolve().parents[1] / "docs" / "examples" / "student_request.json"


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_health_and_prediction_match_saved_pipeline(self) -> None:
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            response = client.post("/predict", json=self.payload)
            self.assertEqual(response.status_code, 200)
            frame = pd.DataFrame([self.payload], columns=list(FEATURES))
            expected = float(app.state.model.predict(frame)[0])
            self.assertAlmostEqual(response.json()["predicted_grade"], expected)

    def test_invalid_requests_return_422(self) -> None:
        invalid = [
            {key: value for key, value in self.payload.items() if key != "studytime"},
            {**self.payload, "studytime": 9},
            {**self.payload, "schoolsup": "maybe"},
            {**self.payload, "G1": 10},
            {**self.payload, "Medu": "2"},
        ]
        with TestClient(app) as client:
            for body in invalid:
                with self.subTest(body=body):
                    response = client.post("/predict", json=body)
                    self.assertEqual(response.status_code, 422)

    def test_missing_model_fails_at_startup(self) -> None:
        with patch("src.api.main.joblib.load", side_effect=FileNotFoundError("missing model")):
            with self.assertRaises(FileNotFoundError):
                with TestClient(app):
                    pass


if __name__ == "__main__":
    unittest.main()
