import unittest

import numpy as np
import pandas as pd

from pipelines.pipeline2_prediction.orchestration.prediction_orchestrator import (
    _held_out_permutation_importance,
)


class _ThresholdModel:
    def predict(self, frame):
        return (np.asarray(frame["signal"]) > 0).astype(int)


class _Package:
    @staticmethod
    def predict_model(model, frame):
        return model.predict(frame)


class FeatureImportanceTests(unittest.TestCase):
    def test_permutation_importance_uses_held_out_source_columns(self):
        signal = np.array([-1, 1] * 30)
        frame = pd.DataFrame({"signal": signal, "noise": np.arange(60)})
        target = (signal > 0).astype(int)
        result = _held_out_permutation_importance(
            package=_Package,
            model=_ThresholdModel(),
            preprocessor=None,
            task_type="classification",
            X_test=frame,
            y_test=target,
            baseline_predictions=target,
            random_state=42,
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["evaluation_scope"], "held_out_test_split")
        self.assertEqual(result["features"][0]["feature"], "signal")
        self.assertGreater(result["features"][0]["importance"], result["features"][1]["importance"])


if __name__ == "__main__":
    unittest.main()
