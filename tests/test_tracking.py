"""Ensure MLflow records candidates without leaking test metrics to all runs."""

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.models.compare import main


class TrackingTests(unittest.TestCase):
    def test_three_runs_and_selected_test_metrics(self) -> None:
        tracking = Mock()
        run_metrics = {}
        artifacts = []
        current_run = None

        @contextmanager
        def start_run(run_name=None, run_id=None):
            nonlocal current_run
            current_run = run_id or run_name
            run_metrics.setdefault(current_run, {})
            yield SimpleNamespace(info=SimpleNamespace(run_id=current_run))
            current_run = None

        def log_metrics(metrics):
            run_metrics[current_run].update(metrics)

        def log_artifact(path, artifact_path=None):
            self.assertTrue(Path(path).is_file())
            artifacts.append((current_run, artifact_path))

        tracking.start_run.side_effect = start_run
        tracking.log_metrics.side_effect = log_metrics
        tracking.log_artifact.side_effect = log_artifact
        with patch.dict(sys.modules, {"mlflow": tracking}), patch(
            "src.models.compare.save_and_verify"
        ):
            main(track=True)

        self.assertEqual(len(run_metrics), 3)
        self.assertEqual(len(artifacts), 3)
        self.assertTrue(all("cv_mae" in metrics for metrics in run_metrics.values()))
        self.assertEqual(
            [name for name, metrics in run_metrics.items() if "test_mae" in metrics],
            ["ridge"],
        )
        self.assertEqual(set(run_metrics["ridge"]), {
            "cv_mae", "cv_mae_std", "test_mae", "test_rmse", "test_r2"
        })


if __name__ == "__main__":
    unittest.main()
