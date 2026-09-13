import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from benchmarks.tabarena.pipeline2_benchmark_runner import (
    Pipeline1PrivacyClearanceError,
    _score,
    load_privacy_cleared_dataset,
    run_evaluations,
)
from pipelines.pipeline2_prediction.models.artifact_io import training_artifact_dir


class BenchmarkRunnerTests(unittest.TestCase):
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
            "task_type": "Supervised Classification",
            "target_column": "target",
            "csv_file": "does-not-exist.csv",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = run_evaluations(
                [dataset],
                root / "runner",
                root / "pipeline1",
                run_id="test",
            )
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["status"] == "failure" for row in rows))
        self.assertTrue(
            all(row["outcome_reason"].startswith("dataset_setup:") for row in rows)
        )

    def test_raw_dataset_without_pipeline1_privacy_clearance_is_rejected(self):
        dataset = {
            "task_id": "raw-only",
            "dataset_name": "raw-only",
            "task_type": "Supervised Classification",
            "target_column": "target",
            "csv_file": "raw.csv",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({
                "email": ["person@example.com"],
                "target": [1],
            }).to_csv(root / "raw.csv", index=False)

            with self.assertRaisesRegex(
                Pipeline1PrivacyClearanceError,
                "Raw-input fallback is prohibited",
            ):
                load_privacy_cleared_dataset(dataset, root / "pipeline1")

    def test_successful_cleaning_without_privacy_stage_is_rejected(self):
        dataset = {
            "task_id": "cleaned-only",
            "dataset_name": "cleaned-only",
            "task_type": "Supervised Classification",
            "target_column": "target",
            "csv_file": "raw.csv",
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "pipeline1" / "cleaned-only"
            artifact_dir.mkdir(parents=True)
            pd.DataFrame({"feature": [1], "target": [0]}).to_csv(
                artifact_dir / "cleaned_dataset.csv",
                index=False,
            )
            (artifact_dir / "cleaning_report.json").write_text(
                json.dumps({
                    "status": "success",
                    "original_dataset": {"file_name": "raw.csv"},
                    "validation": {"passed": True},
                    "decision_summary": {},
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                Pipeline1PrivacyClearanceError,
                "has no Pipeline 1 privacy audit",
            ):
                load_privacy_cleared_dataset(dataset, Path(directory) / "pipeline1")


if __name__ == "__main__":
    unittest.main()
