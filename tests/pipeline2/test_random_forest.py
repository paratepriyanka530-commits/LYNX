import unittest
import warnings

import numpy as np
import pandas as pd

from pipelines.pipeline2_prediction.models.random_forest import (
    build_model,
    get_model_info,
    load_trained_model,
    predict_model,
    predict_with_artifact,
    train_model,
)


class RandomForestTests(unittest.TestCase):
    def test_common_interface_supports_both_task_types(self):
        self.assertEqual(
            get_model_info()["supported_tasks"],
            ["classification", "regression"],
        )
        self.assertEqual(build_model("classification").named_steps["model"].n_estimators, 200)
        self.assertEqual(build_model("regression").named_steps["model"].n_estimators, 200)

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

    def test_supplied_artifact_loads_without_warning_and_predicts(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, metadata = load_trained_model()
        self.assertEqual(caught, [])
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
