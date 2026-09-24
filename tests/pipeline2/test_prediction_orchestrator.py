import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipelines.pipeline2_prediction.orchestration import prediction_orchestrator as po


SEMANTIC_PROFILE = {
    "probable_domain": "testing",
    "subdomain": "synthetic tabular data",
    "dataset_purpose": "binary outcome prediction",
    "row_entity": "synthetic observation",
    "target_concept": "binary outcome",
    "feature_context": ["numeric measurements", "group label"],
    "semantic_summary": "Synthetic observations used to test binary prediction.",
    "domain_confidence": 0.9,
}


class FakeSelector:
    representation = "D"

    def __init__(self, first="random_forest", second="xgboost"):
        if second == first:
            second = next(model_id for model_id in po.CANONICAL_MODEL_IDS if model_id != first)
        self.first = first
        self.second = second
        self.calls = []

    def predict_ranking(self, **kwargs):
        self.calls.append(kwargs)
        ordered = [self.first, self.second] + [
            model_id for model_id in po.CANONICAL_MODEL_IDS
            if model_id not in {self.first, self.second}
        ]
        return [
            {
                "rank": rank,
                "model_name": model_id,
                "model_family": "test_family",
                "predicted_suitability_score": 1.0 - rank / 20.0,
            }
            for rank, model_id in enumerate(ordered, 1)
        ]


class FakeModel:
    def __init__(self, fail_fit=False, poor_predictions=False):
        self.fail_fit = fail_fit
        self.poor_predictions = poor_predictions
        self.classes_ = np.array([0, 1])
        self.training_columns = None
        self.training_indices = None

    def predict_proba(self, X):
        probability = np.full(len(X), 0.1 if self.poor_predictions else 0.6)
        return np.column_stack([1.0 - probability, probability])


class FakePackage:
    def __init__(self, model_id, fail_models=None, poor_models=None, built=None):
        self.model_id = model_id
        self.fail_models = set(fail_models or [])
        self.poor_models = set(poor_models or [])
        self.built = built if built is not None else []

    def build_model(self, task_type, random_state=42):
        model = FakeModel(
            fail_fit=self.model_id in self.fail_models,
            poor_predictions=self.model_id in self.poor_models,
        )
        self.built.append((self.model_id, model))
        return model

    @staticmethod
    def train_model(model, X_train, y_train):
        model.training_columns = list(X_train.columns)
        model.training_indices = list(X_train.index)
        if model.fail_fit:
            raise RuntimeError("controlled fit failure")
        return model

    @staticmethod
    def predict_model(model, X_test):
        if model.poor_predictions:
            return np.zeros(len(X_test), dtype=int)
        return np.asarray([index % 2 for index in range(len(X_test))], dtype=int)


class SpyPreprocessor:
    def __init__(self):
        self.fit_indices = None
        self.transform_indices = None

    def fit_transform(self, frame):
        self.fit_indices = list(frame.index)
        return frame.copy()

    def transform(self, frame):
        self.transform_indices = list(frame.index)
        return frame.copy()


class PredictionOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "feature_a": np.arange(60, dtype=float),
            "feature_b": np.tile([10.0, 20.0, 30.0], 20),
            "target": np.tile(["no", "yes"], 30),
        })

    def _run_fake(
        self,
        *,
        first="random_forest",
        second="xgboost",
        fail_models=(),
        poor_models=(),
        preprocessor=None,
    ):
        selector = FakeSelector(first, second)
        built = []

        def import_package(model_id):
            return FakePackage(
                model_id, fail_models, poor_models, built
            )

        with (
            patch.object(po, "_load_frozen_selector", return_value=(selector, "a" * 64)),
            patch.object(po, "_import_model_package", side_effect=import_package),
            patch.object(po, "_build_external_preprocessor", return_value=preprocessor),
        ):
            output = po.orchestrate_prediction(
                self.frame,
                target_column="target",
                task_type="classification",
                dataset_name="orchestration_unit_test",
                semantic_profile=SEMANTIC_PROFILE,
                split_seed=7,
            )
        return output, selector, built

    def test_target_excluded_split_precedes_preprocessing_and_target_stats_are_train_only(self):
        spy = SpyPreprocessor()
        output, selector, built = self._run_fake(first="xgboost", preprocessor=spy)
        model = built[0][1]
        self.assertNotIn("target", model.training_columns)
        self.assertEqual(spy.fit_indices, output.result["split"]["train_indices"])
        self.assertEqual(spy.transform_indices, output.result["split"]["test_indices"])
        self.assertTrue(set(spy.fit_indices).isdisjoint(spy.transform_indices))
        y_train = pd.Series(output.runtime.y_train)
        expected_ratio = float(y_train.value_counts(normalize=True).min())
        selector_target = selector.calls[0]["target_info"]
        self.assertEqual(selector_target["n_classes"], y_train.nunique())
        self.assertAlmostEqual(selector_target["minority_class_ratio"], expected_ratio)

    def test_frozen_selector_load_is_read_only_and_hash_verified(self):
        path = po.SELECTOR_PATH
        mtime = path.stat().st_mtime_ns
        selector, digest = po._load_frozen_selector(path, verify_hash=True)
        self.assertEqual(digest, po.FROZEN_SELECTOR_SHA256)
        self.assertEqual(selector.representation, "D")
        self.assertEqual(path.stat().st_mtime_ns, mtime)

    def test_top1_success_does_not_execute_top2(self):
        output, _selector, built = self._run_fake()
        self.assertEqual([model_id for model_id, _ in built], ["random_forest"])
        self.assertEqual(len(output.result["execution"]["attempts"]), 1)
        self.assertFalse(output.result["execution"]["fallback_used"])

    def test_top1_technical_failure_executes_top2_once(self):
        output, _selector, built = self._run_fake(fail_models={"random_forest"})
        self.assertEqual(
            [model_id for model_id, _ in built], ["random_forest", "xgboost"]
        )
        attempts = output.result["execution"]["attempts"]
        self.assertEqual([attempt["status"] for attempt in attempts], ["failed", "success"])
        self.assertEqual(attempts[0]["failure_stage"], "model_fit")
        self.assertIsNotNone(attempts[0]["train_runtime_seconds"])
        self.assertTrue(output.result["execution"]["fallback_used"])
        self.assertEqual(output.result["execution"]["selected_rank"], 2)

    def test_poor_metric_does_not_trigger_fallback(self):
        output, _selector, built = self._run_fake(poor_models={"random_forest"})
        self.assertEqual([model_id for model_id, _ in built], ["random_forest"])
        self.assertEqual(len(output.result["execution"]["attempts"]), 1)
        self.assertFalse(output.result["execution"]["fallback_used"])
        self.assertLessEqual(
            output.result["evaluation"]["predictive_metrics"]["balanced_accuracy"],
            0.5,
        )

    def test_both_failures_return_both_attempt_records(self):
        output, _selector, built = self._run_fake(
            fail_models={"random_forest", "xgboost"}
        )
        self.assertEqual(len(built), 2)
        self.assertEqual(output.result["execution"]["status"], "failed")
        self.assertIsNone(output.result["execution"]["selected_model"])
        self.assertEqual(
            [attempt["status"] for attempt in output.result["execution"]["attempts"]],
            ["failed", "failed"],
        )
        self.assertEqual(len(output.result["errors"]), 2)

    def test_result_contract_contains_visualization_and_xai_fields(self):
        output, _selector, _built = self._run_fake()
        payload = output.to_dict()
        self.assertEqual(
            set(payload),
            {
                "schema_version", "run_id", "dataset", "split", "selector",
                "execution", "predictions", "evaluation", "artifacts",
                "warnings", "errors",
            },
        )
        self.assertEqual(len(payload["selector"]["ranking"]), 10)
        self.assertIsNotNone(payload["predictions"]["y_true"])
        self.assertIsNotNone(payload["predictions"]["y_pred"])
        self.assertIn("probabilities", payload["predictions"])
        self.assertIsNotNone(output.runtime.fitted_model)
        self.assertIsNotNone(output.runtime.X_train)
        json.dumps(payload, allow_nan=False)

    def test_random_forest_lightweight_end_to_end(self):
        selector = FakeSelector(first="random_forest", second="xgboost")
        real_package = po._import_model_package("random_forest")

        class SingleThreadRandomForestPackage:
            @staticmethod
            def build_model(task_type, random_state=42):
                model = real_package.build_model(task_type, random_state)
                # The Windows Store Python sandbox denies creation of joblib's
                # worker synchronization pipe. This keeps the real estimator
                # and preprocessing pipeline while making the test single-threaded.
                model.named_steps["model"].set_params(n_jobs=1)
                return model

            train_model = staticmethod(real_package.train_model)
            predict_model = staticmethod(real_package.predict_model)

        with (
            patch.object(po, "_load_frozen_selector", return_value=(selector, "a" * 64)),
            patch.object(
                po, "_import_model_package", return_value=SingleThreadRandomForestPackage
            ),
        ):
            output = po.orchestrate_prediction(
                self.frame,
                target_column="target",
                task_type="classification",
                dataset_name="random_forest_end_to_end",
                semantic_profile=SEMANTIC_PROFILE,
                split_seed=11,
            )
        self.assertEqual(output.result["execution"]["status"], "success")
        self.assertEqual(output.result["execution"]["selected_model"], "random_forest")
        self.assertEqual(len(output.result["execution"]["attempts"]), 1)
        self.assertEqual(
            type(output.runtime.fitted_model).__name__, "Pipeline"
        )


if __name__ == "__main__":
    unittest.main()
