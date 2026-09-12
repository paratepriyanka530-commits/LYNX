import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


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

    @unittest.skipUnless(importlib.util.find_spec("tabicl"), "tabicl is not installed")
    def test_tabiclv2_reuses_train_fitted_preprocessor(self):
        from pipelines.pipeline2_prediction.models import tabiclv2

        self._assert_train_only_preprocessor(tabiclv2, "TabICLClassifier")
        self.assertEqual(
            tabiclv2.get_model_info()["supported_tasks"],
            ["classification", "regression"],
        )

    @unittest.skipUnless(importlib.util.find_spec("pytabkit"), "pytabkit is not installed")
    def test_ft_transformer_reuses_train_fitted_preprocessor(self):
        from pipelines.pipeline2_prediction.models import ft_transformer

        self._assert_train_only_preprocessor(ft_transformer, "FTT_D_Classifier")
        info = ft_transformer.get_model_info()
        self.assertEqual(info["supported_tasks"], ["classification", "regression"])
        self.assertIn("non-pretrained", info["purpose"])


if __name__ == "__main__":
    unittest.main()
