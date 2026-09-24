import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.tabarena.pipeline2_benchmark_runner import _base_record
from benchmarks.tabarena.pipeline2_groundwork import (
    EVALUATION_MODEL_IDS,
    crumb_feasibility,
    validate_log_record,
)
from pipelines.pipeline2_prediction.models.crumb import CrumbConfig, CrumbModel


class _RecordingBase:
    def __init__(self):
        self.fit_context_sizes = []
        self.seen_targets = []
        self.value = 0


def _train(model, X, y):
    model.fit_context_sizes.append(len(X))
    model.seen_targets.extend(np.asarray(y).tolist())
    model.value = int(np.asarray(y)[0])
    return model


def _predict(model, X):
    return np.repeat(model.value, len(X))


class CrumbTests(unittest.TestCase):
    def test_optional_wrappers_expose_common_interface_without_joining_roster(self):
        from pipelines.pipeline2_prediction.models import tabiclv2_crumb, tabpfn_crumb

        for package in (tabpfn_crumb, tabiclv2_crumb):
            for name in ("get_model_info", "build_model", "train_model", "predict_model"):
                self.assertTrue(callable(getattr(package, name)))
            self.assertTrue(package.get_model_info()["experimental"])

    def _run(self):
        X_train = pd.DataFrame({
            "x": np.arange(40, dtype=float),
            "z": np.arange(40, dtype=float) % 7,
        })
        y_train = pd.Series(np.arange(40) % 2)
        X_test = pd.DataFrame({
            "x": [1.5, 2.5, 20.5, 21.5, 37.5, 38.5],
            "z": [1.0, 2.0, 6.0, 0.0, 2.0, 3.0],
        })
        base = _RecordingBase()
        model = CrumbModel(
            base,
            _train,
            _predict,
            CrumbConfig(
                n_clusters=3,
                context_fraction=0.5,
                max_context_size=7,
                rff_dimension=16,
                greedy_batch_size=2,
                random_state=11,
                early_stopping=False,
            ),
        ).fit(X_train, y_train)
        predictions = model.predict(X_test)
        return model, base, predictions, X_train, y_train

    def test_selection_is_sized_valid_deterministic_and_predictions_restore_order(self):
        first, base, predictions, X_train, _ = self._run()
        second, _, second_predictions, _, _ = self._run()
        self.assertEqual(predictions.shape, (6,))
        self.assertEqual(predictions.tolist(), second_predictions.tolist())
        self.assertEqual(first.last_selection_, second.last_selection_)
        self.assertTrue(all(size == 7 for size in base.fit_context_sizes))
        for cluster in first.last_selection_:
            self.assertEqual(len(cluster["context_indices"]), 7)
            self.assertTrue(all(0 <= index < len(X_train) for index in cluster["context_indices"]))

    def test_no_test_labels_exist_and_train_fitted_scaling_is_train_only(self):
        model, base, _, X_train, y_train = self._run()
        self.assertEqual(set(base.seen_targets), set(y_train.tolist()))
        np.testing.assert_allclose(model.scaler_.mean_, X_train.mean().to_numpy())
        with self.assertRaises(TypeError):
            model.predict(pd.DataFrame({"x": [1.0], "z": ["not-numeric"]}))

    def test_base_and_crumb_feasibility_are_separate(self):
        result = crumb_feasibility("tabpfn", 20_000, 20, "classification", 2)
        self.assertFalse(result["base_model_feasible"])
        self.assertTrue(result["crumb_variant_feasible"])
        self.assertTrue(result["changes_feasibility"])
        self.assertEqual(result["context_size"], 2_000)
        feature_blocked = crumb_feasibility("tabpfn", 20_000, 501, "classification", 2)
        self.assertFalse(feature_blocked["crumb_variant_feasible"])

    def test_variant_logging_is_distinct_and_canonical_roster_unchanged(self):
        self.assertEqual(len(EVALUATION_MODEL_IDS), 10)
        self.assertNotIn("tabpfn_crumb", EVALUATION_MODEL_IDS)
        dataset = {"task_id": "x", "target_column": "y"}
        row = _base_record(
            dataset, "tabpfn_crumb", "classification", {}, None, 42,
            "tabpfn", "crumb",
        )
        row.update({"feasible": False, "status": "infeasible_skip"})
        validate_log_record(row)
        self.assertEqual(row["base_model_id"], "tabpfn")
        self.assertEqual(row["execution_strategy"], "crumb")


if __name__ == "__main__":
    unittest.main()
