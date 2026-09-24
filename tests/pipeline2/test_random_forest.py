import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from pipelines.pipeline2_prediction.models.random_forest import (
    build_model,
    get_model_info,
    load_trained_model,
    predict_model,
    predict_with_artifact,
    train_model,
)
from benchmarks.tabarena.pipeline2_benchmark_runner import _optional_oob_result


class RandomForestTests(unittest.TestCase):
    def test_common_interface_supports_both_task_types(self):
        self.assertEqual(
            get_model_info()["supported_tasks"],
            ["classification", "regression"],
        )
        self.assertEqual(build_model("classification").named_steps["model"].n_estimators, 200)
        self.assertEqual(build_model("regression").named_steps["model"].n_estimators, 200)
        self.assertFalse(build_model("classification").named_steps["model"].oob_score)
        self.assertTrue(build_model("classification").named_steps["model"].bootstrap)

    def test_optional_oob_is_training_only_and_available(self):
        features = pd.DataFrame({
            "numeric": np.arange(120),
            "category": np.where(np.arange(120) % 3, "a", "b"),
        })
        target = np.arange(120) % 2
        model = build_model("classification", random_state=7, oob_score=True)
        train_model(model, features, target)
        score, reason = _optional_oob_result("random_forest", model, True)
        self.assertIsNone(reason)
        self.assertTrue(np.isfinite(score))
        self.assertEqual(_optional_oob_result("random_forest", model, False), (None, None))


    def test_mixed_type_preprocessor_is_fitted_once_and_reused(self):
        features = pd.DataFrame({
            "numeric": [1, 2, 3, 4, 5, 6, 7, 8],
            "category": ["a", "a", "b", "b", "a", "b", "a", "b"],
        })
        target = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        model = train_model(build_model("classification", 7), features, target)
        preprocessor_id = id(model.named_steps["preprocessor"])

        predictions = predict_model(
            model,
            pd.DataFrame({"numeric": [9, 10], "category": ["a", "unseen"]}),
        )

        self.assertEqual(predictions.shape, (2,))
        self.assertEqual(preprocessor_id, id(model.named_steps["preprocessor"]))
        categories = (
            model.named_steps["preprocessor"]
            .named_transformers_["categorical"]
            .categories_[0]
            .tolist()
        )
        self.assertEqual(categories, ["a", "b"])

    def test_supplied_artifact_loads_with_only_documented_version_warnings(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, metadata = load_trained_model()
        self.assertEqual(len(caught), 207)
        self.assertTrue(
            all(item.category is InconsistentVersionWarning for item in caught)
        )
        self.assertEqual(type(model.named_steps["model"]).__name__, "RandomForestRegressor")
        self.assertEqual(len(model.named_steps["model"].estimators_), 200)

        row = {
            "pizza_id": 1,
            "order_id": 1,
            "pizza_name_id": "test",
            "quantity": 1,
            "order_date": "2015-01-01",
            "order_time": "12:00:00",
            "total_price": 12.0,
            "pizza_size": "M",
            "pizza_category": "Classic",
            "pizza_ingredients": "Cheese",
            "pizza_name": "Test Pizza",
        }
        predictions = predict_with_artifact(pd.DataFrame([row, row]))
        self.assertEqual(predictions.shape, (2,))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertEqual(metadata["problem_type"], "regression")


if __name__ == "__main__":
    unittest.main()
