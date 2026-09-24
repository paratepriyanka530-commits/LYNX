import json
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss as sklearn_log_loss

from benchmarks.tabarena.evaluation_metrics import (
    ECE_METHOD,
    ECE_N_BINS,
    PROBABILITY_CAPABILITIES,
    brier_score,
    classification_results,
    expected_calibration_error,
    regression_results,
    validate_probabilities,
)
from benchmarks.tabarena.pipeline2_groundwork import (
    EVALUATION_MODEL_IDS,
    LOG_COLUMNS,
    initialize_log,
    validate_log_record,
)
from pipelines.pipeline2_prediction.models import random_forest


class _ProbabilityModel:
    def __init__(self, probabilities, classes=(0, 1)):
        self.probabilities = np.asarray(probabilities)
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X):
        return self.probabilities


class _BrokenProbabilityModel:
    classes_ = np.asarray([0, 1])

    def predict_probability(self, X):
        raise RuntimeError("controlled probability failure")


class PhaseEMetricTests(unittest.TestCase):
    def test_all_ten_models_have_audited_probability_support(self):
        self.assertEqual(set(PROBABILITY_CAPABILITIES), set(EVALUATION_MODEL_IDS))
        for capability in PROBABILITY_CAPABILITIES.values():
            self.assertTrue(capability["supports_classification_probabilities"])
            self.assertTrue(capability["probability_source"])
            self.assertTrue(capability["probability_class_order_available"])

    def test_perfect_binary_brier_and_ece(self):
        y = np.asarray([0, 1, 0, 1])
        probabilities = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
        classes = np.asarray([0, 1])
        self.assertEqual(brier_score(y, probabilities, classes), 0.0)
        self.assertEqual(expected_calibration_error(y, probabilities, classes), 0.0)
        self.assertEqual(
            brier_score(y, probabilities, classes),
            brier_score_loss(y, probabilities[:, 1]),
        )

    def test_deliberately_poor_binary_probabilities(self):
        y = np.asarray([0, 1])
        probabilities = np.asarray([[0.1, 0.9], [0.9, 0.1]])
        classes = np.asarray([0, 1])
        self.assertAlmostEqual(brier_score(y, probabilities, classes), 0.81)
        self.assertAlmostEqual(expected_calibration_error(y, probabilities, classes), 0.9)

    def test_multiclass_brier_is_unnormalized_class_sum(self):
        y = np.asarray([0, 2])
        probabilities = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
        expected = ((0.3**2 + 0.2**2 + 0.1**2) * 2) / 2
        self.assertAlmostEqual(brier_score(y, probabilities, np.asarray([0, 1, 2])), expected)

    def test_multiclass_ece(self):
        y = np.asarray([0, 1, 2, 0])
        probabilities = np.asarray([
            [0.8, 0.1, 0.1], [0.1, 0.7, 0.2],
            [0.2, 0.5, 0.3], [0.4, 0.3, 0.3],
        ])
        self.assertAlmostEqual(
            expected_calibration_error(y, probabilities, np.asarray([0, 1, 2])),
            0.4,
        )

    def test_ece_is_frozen_at_fifteen_bins(self):
        with self.assertRaisesRegex(ValueError, "exactly 15"):
            expected_calibration_error(
                [0], np.asarray([[0.8, 0.2]]), np.asarray([0, 1]), n_bins=10
            )
        self.assertEqual(ECE_N_BINS, 15)
        self.assertEqual(ECE_METHOD, "top_label_equal_width")

    def test_ece_bin_boundary_is_left_closed_and_one_is_last_bin(self):
        below = np.nextafter(0.6, 0.0)
        probabilities = np.asarray([[0.6, 0.4], [1.0 - below, below], [1.0, 0.0]])
        y = np.asarray([0, 0, 0])
        expected = (abs(1.0 - 0.6) + abs(0.0 - below) + abs(1.0 - 1.0)) / 3
        self.assertAlmostEqual(
            expected_calibration_error(y, probabilities, np.asarray([0, 1])), expected
        )

    def test_probability_row_sum_validation(self):
        _, _, reason = validate_probabilities(
            [[0.2, 0.7]], [0, 1], [0], [0, 1]
        )
        self.assertIn("sum to 1", reason)

    def test_probability_nan_and_inf_rejection(self):
        for value in (np.nan, np.inf):
            _, _, reason = validate_probabilities(
                [[value, 1.0]], [0, 1], [0], [0, 1]
            )
            self.assertIn("NaN or infinite", reason)

    def test_probability_class_order_mismatch(self):
        _, _, reason = validate_probabilities(
            [[0.2, 0.3, 0.5]], [0, 1, 2], [0], [0, 1]
        )
        self.assertIn("expected fitted classes", reason)

    def test_unavailable_probability_handling_and_failure_isolation(self):
        predictive, calibration, metadata = classification_results(
            [0, 1], [0, 1], [0, 1], "tabpfn", _BrokenProbabilityModel(), [[1], [2]]
        )
        self.assertEqual(predictive["accuracy"], 1.0)
        self.assertIsNone(predictive["roc_auc"])
        self.assertIsNone(calibration["brier_score"])
        self.assertEqual(calibration["status"], "unavailable")
        self.assertIn("controlled probability failure", metadata["reason"])

    def test_regression_calibration_is_not_applicable(self):
        predictive, calibration, metadata = regression_results([1, 2, 3], [1, 2, 3])
        self.assertEqual(predictive["rmse"], 0.0)
        self.assertEqual(calibration["status"], "not_applicable")
        self.assertEqual(calibration["reason"], "point-prediction regression protocol")
        self.assertEqual(metadata["status"], "not_applicable")

    def test_schema_round_trip_and_legacy_record_compatibility(self):
        record = {
            "dataset_id": "legacy", "model_id": "random_forest",
            "target_column": "y", "task_type": "classification", "feasible": True,
            "status": "success", "outcome_reason": None,
            "metric_name": "balanced_accuracy", "metric_value": 1.0,
            "baseline_relative_score": 1.0, "train_runtime_seconds": 1.0,
            "inference_runtime_seconds": 0.1, "runtime_seconds": 1.1,
            "oob_score": None, "oob_diagnostic_reason": None,
            "split_seed": 42, "hyperparams": {}, "semantic_meta_features": {},
            "statistical_meta_features": {}, "source": "team-original-run",
        }
        validate_log_record(record)
        restored = json.loads(json.dumps(record))
        self.assertEqual(restored["execution_strategy"], "standard")
        self.assertEqual(restored["base_model_id"], "random_forest")
        self.assertEqual(restored["calibration_metrics"]["status"], "unavailable")

    def test_populated_legacy_csv_is_additively_migrated(self):
        old_columns = [field for field in LOG_COLUMNS if field not in {
            "predictive_metrics", "calibration_metrics", "probability_metadata"
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_run_log.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=old_columns)
                writer.writeheader()
                writer.writerow({field: "kept" for field in old_columns})
            initialize_log(Path(directory))
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0]), LOG_COLUMNS)
        self.assertEqual(rows[0]["dataset_id"], "kept")
        self.assertEqual(rows[0]["calibration_metrics"], "")

    def test_crumb_metadata_remains_schema_compatible(self):
        record = {
            "dataset_id": "crumb", "model_id": "tabpfn_crumb",
            "base_model_id": "tabpfn", "execution_strategy": "crumb",
            "target_column": "y", "task_type": "classification", "feasible": False,
            "status": "infeasible_skip", "outcome_reason": "features",
            "metric_name": None, "metric_value": None, "baseline_relative_score": None,
            "train_runtime_seconds": None, "inference_runtime_seconds": None,
            "runtime_seconds": None, "oob_score": None, "oob_diagnostic_reason": None,
            "split_seed": 42, "hyperparams": {}, "semantic_meta_features": {},
            "statistical_meta_features": {}, "source": "team-original-run",
        }
        validate_log_record(record)
        self.assertEqual(record["execution_strategy"], "crumb")

    def test_small_binary_multiclass_and_regression_lifecycles(self):
        X = pd.DataFrame({"x": np.arange(60), "kind": np.tile(["a", "b", "c"], 20)})
        for classes in (2, 3):
            y = np.arange(60) % classes
            model = random_forest.build_model("classification", random_state=4)
            random_forest.train_model(model, X.iloc[:45], y[:45])
            predictions = random_forest.predict_model(model, X.iloc[45:])
            predictive, calibration, metadata = classification_results(
                y[:45], y[45:], predictions, "random_forest", model, X.iloc[45:]
            )
            self.assertEqual(calibration["status"], "available")
            self.assertAlmostEqual(
                predictive["log_loss"],
                sklearn_log_loss(y[45:], model.predict_proba(X.iloc[45:]), labels=model.classes_),
            )
            self.assertEqual(metadata["class_order"], model.classes_.tolist())
            restored = json.loads(json.dumps({
                "predictive_metrics": predictive,
                "calibration_metrics": calibration,
                "probability_metadata": metadata,
            }))
            self.assertEqual(restored["calibration_metrics"]["ece_n_bins"], 15)
        regression_model = random_forest.build_model("regression", random_state=4)
        target = np.linspace(0, 1, 60)
        random_forest.train_model(regression_model, X.iloc[:45], target[:45])
        regression_predictions = random_forest.predict_model(regression_model, X.iloc[45:])
        predictive, calibration, _ = regression_results(target[45:], regression_predictions)
        self.assertIn("mae", predictive)
        self.assertEqual(calibration["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
