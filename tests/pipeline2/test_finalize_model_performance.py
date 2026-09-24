import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.tabarena.finalize_model_performance import finalize


class FinalizeModelPerformanceTests(unittest.TestCase):
    def test_converts_checkpoint_without_encoding_failure_as_zero(self):
        row = {
            "dataset_id": "363621", "model_id": "xgboost",
            "base_model_id": "xgboost", "execution_strategy": "standard",
            "run_id": "test", "target_column": "target",
            "task_type": "classification", "feasible": True,
            "status": "failure", "outcome_reason": "RuntimeError: controlled",
            "infeasible_reason": None,
            "failure": {"stage": "train_or_evaluate", "exception_type": "RuntimeError", "message": "controlled"},
            "split": {"train_rows": 80, "test_rows": 20, "test_size": 0.2, "split_seed": 42, "stratified": True},
            "metric_name": None, "metric_value": None,
            "baseline_relative_score": None, "train_runtime_seconds": None,
            "inference_runtime_seconds": None, "runtime_seconds": None,
            "oob_score": None, "oob_diagnostic_reason": None,
            "split_seed": 42, "hyperparams": {}, "predictive_metrics": None,
            "calibration_metrics": None, "probability_metadata": None,
            "semantic_meta_features": {}, "statistical_meta_features": {},
            "resource_usage": None, "source": "team-original-run",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "rows.json"
            output = root / "performance.json"
            checkpoint.write_text(json.dumps([row]), encoding="utf-8")
            payload = finalize(checkpoint, output, expected_count=1)
            saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["validation"]["failure_count"], 1)
        self.assertIsNone(saved["records"][0]["primary_metric"]["value"])
        self.assertEqual(saved["records"][0]["failure"]["stage"], "train_or_evaluate")


if __name__ == "__main__":
    unittest.main()
