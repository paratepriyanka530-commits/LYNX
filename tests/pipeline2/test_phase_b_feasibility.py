import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.pipeline2_prediction.models import xgboost
from pipelines.pipeline2_prediction.models.xgboost.xgboost_preprocessor import (
    XGBoostPreprocessor,
)
from benchmarks.tabarena.diagnostics.xtab_feature_capacity_probe import (
    run_capacity_probe,
)


class _FakeTrainer:
    device = "cpu"


class _FakeXTabModel:
    trainer = _FakeTrainer()


class _FakeXTabPackage:
    @staticmethod
    def build_model(task_type, random_state=42, **overrides):
        return _FakeXTabModel()

    @staticmethod
    def train_model(model, features, target):
        return model

    @staticmethod
    def predict_model(model, features):
        return np.zeros(len(features), dtype=int)


class XTabCapacityProbeTests(unittest.TestCase):
    def test_probe_persists_empirical_results_without_a_hard_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = run_capacity_probe(
                Path(directory),
                feature_counts=(20, 50),
                rows=32,
                model_package=_FakeXTabPackage,
            )
            self.assertTrue((Path(directory) / "xtab_feature_capacity.json").is_file())
            self.assertTrue((Path(directory) / "xtab_feature_capacity.csv").is_file())
        self.assertEqual(payload["classification"], "EMPIRICAL_SAFE_RANGE")
        self.assertEqual(payload["highest_successful_feature_count"], 50)
        self.assertIsNone(payload["hard_maximum_features"])
        self.assertTrue(all(item["prediction_success"] for item in payload["records"]))


class XGBoostMissingValueAuditTests(unittest.TestCase):
    def test_numeric_nans_are_train_median_imputed_before_xgboost(self):
        features = pd.DataFrame({
            "signal": [0.0, np.nan, 0.2, 0.3, np.nan, 0.5, 1.0, 1.1,
                       np.nan, 1.3, 1.4, 1.5],
            "other": [1.0, 1.1, np.nan, 1.3, 1.4, 1.5, 2.0, np.nan,
                      2.2, 2.3, 2.4, 2.5],
        })
        target = pd.Series([0] * 6 + [1] * 6)
        before = int(features.isna().sum().sum())
        preprocessor = XGBoostPreprocessor()
        transformed = preprocessor.fit_transform(features)
        after = int(transformed.isna().sum().sum())

        model = xgboost.build_model("classification", random_state=7)
        observed = {}
        original_fit = model.model.fit

        def recording_fit(X_train, y_train, *args, **kwargs):
            observed["nan_count"] = int(pd.DataFrame(X_train).isna().sum().sum())
            return original_fit(X_train, y_train, *args, **kwargs)

        with patch.object(model.model, "fit", side_effect=recording_fit):
            xgboost.train_model(model, transformed, target)
        predictions = np.asarray(xgboost.predict_model(model, transformed)).reshape(-1)

        self.assertGreater(before, 0)
        self.assertEqual(after, 0)
        self.assertEqual(observed["nan_count"], 0)
        self.assertEqual(predictions.shape, (len(features),))
        self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
