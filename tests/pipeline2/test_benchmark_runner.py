import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
from unittest.mock import patch

from benchmarks.tabarena.pipeline2_benchmark_runner import (
    _model_overrides,
    _optional_oob_result,
    _score,
    run_evaluations,
)
from benchmarks.tabarena.tabarena_benchmark import validate_record
from pipelines.pipeline2_prediction.models.artifact_io import training_artifact_dir


class BenchmarkRunnerTests(unittest.TestCase):
    def test_oob_option_is_random_forest_only(self):
        self.assertEqual(_model_overrides("random_forest", False, False), {})
        self.assertEqual(
            _model_overrides("random_forest", False, True),
            {"oob_score": True},
        )
        self.assertEqual(_model_overrides("xgboost", False, True), {})
        self.assertEqual(_optional_oob_result("xgboost", object(), True), (None, None))

    def test_direct_tabarena_run_is_checkpointed_and_resumed(self):
        dataset = {
            "task_id": "rf-oob",
            "dataset_name": "rf-oob",
            "task_type": "classification",
            "target_column": "target",
            "csv_file": "raw.csv",
        }
        frame = pd.DataFrame({
            "feature": np.arange(120),
            "category": np.where(np.arange(120) % 3, "a", "b"),
            "target": np.arange(120) % 2,
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame.to_csv(root / "raw.csv", index=False)
            output = root / "model_performance.json"
            with patch(
                "benchmarks.tabarena.tabarena_benchmark.CANONICAL_MODEL_IDS",
                ("random_forest",),
            ), patch("benchmarks.tabarena.tabarena_benchmark.DATASET_DIR", root):
                rows = run_evaluations([dataset], output)
                first = output.read_bytes()
                resumed = run_evaluations([dataset], output)
                second = output.read_bytes()
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(resumed), 1)
        self.assertEqual(first, second)
        validate_record(rows[0])
        self.assertEqual(rows[0]["primary_metric_name"], "balanced_accuracy")
        self.assertTrue(np.isfinite(rows[0]["primary_metric_value"]))
        self.assertIn("accuracy", rows[0]["classification_metrics"])
        self.assertIsNone(rows[0]["regression_metrics"])

    def test_training_artifacts_are_isolated_by_dataset_and_run(self):
        with tempfile.TemporaryDirectory() as directory:
            first = training_artifact_dir(
                "xgboost", "dataset/a", "run one", Path(directory)
            )
            second = training_artifact_dir(
                "xgboost", "dataset/b", "run one", Path(directory)
            )
        self.assertNotEqual(first, second)
        self.assertIn("dataset_a", str(first))
        self.assertIn("run_one", str(first))

    def test_classification_baseline_relative_score(self):
        name, metric, relative = _score(
            "classification",
            pd.Series([0, 0, 0, 1]),
            pd.Series([0, 0, 1, 1]),
            pd.Series([0, 0, 1, 1]),
        )
        self.assertEqual(name, "balanced_accuracy")
        self.assertEqual(metric, 1.0)
        self.assertEqual(relative, 1.0)

    def test_regression_baseline_relative_score(self):
        name, metric, relative = _score(
            "regression", [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]
        )
        self.assertEqual(name, "rmse")
        self.assertEqual(metric, 0.0)
        self.assertEqual(relative, 1.0)

    def test_dataset_setup_failure_still_logs_all_candidate_rows(self):
        dataset = {
            "task_id": 999999,
            "dataset_name": "missing-test-dataset",
            "task_type": "classification",
            "target_column": "target",
            "csv_file": "does-not-exist.csv",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = run_evaluations(
                [dataset],
                root / "model_performance.json",
            )
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["feasible"] for row in rows))
        self.assertTrue(
            all(row["infeasible_reason"].startswith("runtime_failure:dataset_setup:") for row in rows)
        )


if __name__ == "__main__":
    unittest.main()
