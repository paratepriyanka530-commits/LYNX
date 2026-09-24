import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


def _tabicl_checkpoints_are_cached():
    if importlib.util.find_spec("tabicl") is None:
        return False
    from huggingface_hub import try_to_load_from_cache

    return all(
        isinstance(try_to_load_from_cache("jingang/TabICL", filename), str)
        for filename in (
            "tabicl-classifier-v2-20260212.ckpt",
            "tabicl-regressor-v2-20260212.ckpt",
        )
    )


class _RecordingEstimator:
    def __init__(self, random_state=None, **kwargs):
        self.random_state = random_state
        self.kwargs = kwargs
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1
        self.fit_columns = tuple(X.columns)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.75, 0.25], (len(X), 1))


class _CheckpointFailureEstimator(_RecordingEstimator):
    def fit(self, X, y):
        raise OSError("checkpoint is not cached and download failed")


class PhaseBCandidateTests(unittest.TestCase):
    def _assert_train_only_preprocessor(self, package, estimator_name):
        train = pd.DataFrame({"number": [1.0, None, 5.0], "label": ["a", None, "b"]})
        test = pd.DataFrame({"number": [None], "label": [None]})
        with patch.object(package, estimator_name, _RecordingEstimator):
            model = package.build_model("classification", random_state=7)
            package.train_model(model, train, pd.Series([0, 1, 0]))
            state_before = (
                model.preprocessor.fit_calls,
                tuple(model.preprocessor.feature_columns),
                dict(model.preprocessor.numeric_fill_values),
            )
            prediction = package.predict_model(model, test)
            probability = model.predict_proba(test)
            state_after = (
                model.preprocessor.fit_calls,
                tuple(model.preprocessor.feature_columns),
                dict(model.preprocessor.numeric_fill_values),
            )
        self.assertEqual(model.fit_calls, 1)
        self.assertEqual(model.estimator.fit_calls, 1)
        self.assertEqual(state_before, state_after)
        self.assertEqual(prediction.shape, (1,))
        self.assertEqual(probability.shape, (1, 2))

    def _assert_regression_path(self, package, estimator_name):
        train = pd.DataFrame({"number": [1.0, None, 5.0]})
        test = pd.DataFrame({"number": [None]})
        with patch.object(package, estimator_name, _RecordingEstimator):
            model = package.build_model("regression", random_state=7)
            package.train_model(model, train, pd.Series([1.0, 2.0, 3.0]))
            state_before = (
                model.preprocessor.fit_calls,
                dict(model.preprocessor.numeric_fill_values),
            )
            prediction = package.predict_model(model, test)
            state_after = (
                model.preprocessor.fit_calls,
                dict(model.preprocessor.numeric_fill_values),
            )
        self.assertEqual(model.fit_calls, 1)
        self.assertEqual(state_before, state_after)
        self.assertEqual(prediction.shape, (1,))
        self.assertTrue(np.isfinite(prediction).all())

    @unittest.skipUnless(importlib.util.find_spec("tabicl"), "tabicl is not installed")
    def test_tabiclv2_reuses_train_fitted_preprocessor(self):
        from pipelines.pipeline2_prediction.models import tabiclv2

        self._assert_train_only_preprocessor(tabiclv2, "TabICLClassifier")
        self.assertEqual(
            tabiclv2.get_model_info()["supported_tasks"],
            ["classification", "regression"],
        )
        self._assert_regression_path(tabiclv2, "TabICLRegressor")
        with self.assertRaisesRegex(ValueError, "Unsupported task"):
            tabiclv2.build_model("clustering")

    @unittest.skipUnless(importlib.util.find_spec("tabicl"), "tabicl is not installed")
    def test_tabiclv2_reports_checkpoint_provisioning_failure(self):
        from pipelines.pipeline2_prediction.models import tabiclv2

        with patch.object(
            tabiclv2,
            "TabICLClassifier",
            _CheckpointFailureEstimator,
        ):
            model = tabiclv2.build_model("classification")
            with self.assertRaisesRegex(
                tabiclv2.TabICLProvisioningError,
                "public jingang/TabICL download",
            ):
                tabiclv2.train_model(
                    model,
                    pd.DataFrame({"x": [0.0, 1.0]}),
                    pd.Series([0, 1]),
                )

    @unittest.skipUnless(
        _tabicl_checkpoints_are_cached(),
        "official TabICLv2 classification/regression checkpoints are not cached",
    )
    def test_tabiclv2_real_classification_and_regression_lifecycles(self):
        import torch
        from pipelines.pipeline2_prediction.models import tabiclv2

        rng = np.random.default_rng(42)
        frame = pd.DataFrame({
            "x1": rng.normal(size=48),
            "x2": rng.normal(size=48),
            "kind": np.where(np.arange(48) % 2, "a", "b"),
        })
        frame.loc[[2, 17], "x1"] = np.nan
        train, test = frame.iloc[:36], frame.iloc[36:]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        targets = {
            "classification": pd.Series(np.arange(36) % 2),
            "regression": pd.Series(rng.normal(size=36)),
        }
        for task_type, target in targets.items():
            model = tabiclv2.build_model(
                task_type,
                random_state=42,
                n_estimators=1,
                batch_size=1,
                device=device,
                allow_auto_download=False,
            )
            fresh = tabiclv2.build_model(
                task_type,
                random_state=42,
                n_estimators=1,
                batch_size=1,
                device=device,
                allow_auto_download=False,
            )
            self.assertIsNot(model.estimator, fresh.estimator)
            tabiclv2.train_model(model, train, target)
            state_before = (
                model.preprocessor.fit_calls,
                dict(model.preprocessor.numeric_fill_values),
            )
            predictions = np.asarray(
                tabiclv2.predict_model(model, test)
            ).reshape(-1)
            state_after = (
                model.preprocessor.fit_calls,
                dict(model.preprocessor.numeric_fill_values),
            )
            self.assertEqual(state_before, state_after)
            self.assertEqual(predictions.shape, (len(test),))
            self.assertTrue(np.isfinite(predictions).all())
            if task_type == "classification":
                probabilities = np.asarray(model.predict_proba(test))
                self.assertEqual(probabilities.shape, (len(test), 2))
                self.assertTrue(np.isfinite(probabilities).all())
                np.testing.assert_allclose(
                    probabilities.sum(axis=1),
                    1.0,
                    atol=1e-6,
                )

    @unittest.skipUnless(importlib.util.find_spec("pytabkit"), "pytabkit is not installed")
    def test_ft_transformer_reuses_train_fitted_preprocessor(self):
        from pipelines.pipeline2_prediction.models import ft_transformer

        self._assert_train_only_preprocessor(ft_transformer, "FTT_D_Classifier")
        info = ft_transformer.get_model_info()
        self.assertEqual(info["supported_tasks"], ["classification", "regression"])
        self.assertIn("non-pretrained", info["purpose"])
        self._assert_regression_path(ft_transformer, "FTT_D_Regressor")
        with self.assertRaisesRegex(ValueError, "Unsupported task"):
            ft_transformer.build_model("clustering")

    def test_new_candidate_artifact_directories_are_isolated(self):
        from pipelines.pipeline2_prediction.models.artifact_io import (
            training_artifact_dir,
        )

        for model_id in ("realmlp", "tabm", "tabiclv2", "ft_transformer"):
            with tempfile.TemporaryDirectory() as directory:
                first = training_artifact_dir(
                    model_id, "dataset-a", "run-a", Path(directory)
                )
                second = training_artifact_dir(
                    model_id, "dataset-b", "run-a", Path(directory)
                )
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
