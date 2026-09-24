import importlib.util
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


PYTABKIT_AVAILABLE = importlib.util.find_spec("pytabkit") is not None
BLOOD_PATH = "benchmarks/tabarena/datasets/blood-transfusion-service-center.csv"
TARGET = "DonatedBloodInMarch2007"


class PytabkitPreprocessorTests(unittest.TestCase):
    def test_train_medians_and_schema_are_reused_without_refitting(self):
        from pipelines.pipeline2_prediction.models.pytabkit_candidate import (
            PytabkitPreprocessor,
        )

        train = pd.DataFrame({"number": [1.0, None, 5.0], "label": ["a", None, "b"]})
        test = pd.DataFrame({"number": [None], "label": [None]})
        preprocessor = PytabkitPreprocessor()
        preprocessor.fit_transform(train)
        transformed = preprocessor.transform(test)
        self.assertEqual(preprocessor.fit_calls, 1)
        self.assertEqual(transformed.loc[0, "number"], 3.0)
        self.assertEqual(transformed.loc[0, "label"], "__LYNX_MISSING__")


@unittest.skipUnless(PYTABKIT_AVAILABLE, "pytabkit is installed in the pinned model environment")
class PytabkitCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = pd.read_csv(BLOOD_PATH)
        X = frame.drop(columns=[TARGET])
        y = frame[TARGET]
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        regression_target = (
            X.iloc[:, 0].astype(float) * 0.25
            + X.iloc[:, 1].astype(float) * 0.5
        )
        (
            cls.X_regression_train,
            cls.X_regression_test,
            cls.y_regression_train,
            cls.y_regression_test,
        ) = train_test_split(
            X,
            regression_target,
            test_size=0.2,
            random_state=42,
        )

    def _assert_lifecycle(self, package_name, task_type):
        package = __import__(
            f"pipelines.pipeline2_prediction.models.{package_name}",
            fromlist=["build_model"],
        )
        if package_name == "ft_transformer":
            config = {
                "device": "cpu",
                "n_threads": 1,
                "max_epochs": 2,
                "es_patience": 1,
                "use_checkpoints": False,
            }
        else:
            config = {
                "device": "cpu",
                "n_threads": 1,
                "n_epochs": 2,
                **({"patience": 1} if package_name == "tabm" else {}),
            }
        if task_type == "classification":
            X_train, X_test = self.X_train, self.X_test
            y_train, y_test = self.y_train, self.y_test
        else:
            X_train, X_test = self.X_regression_train, self.X_regression_test
            y_train, y_test = self.y_regression_train, self.y_regression_test
        model = package.build_model(
            task_type,
            random_state=42,
            **config,
        )
        fresh_model = package.build_model(task_type, random_state=42, **config)
        self.assertIsNot(model, fresh_model)
        self.assertIsNot(model.estimator, fresh_model.estimator)
        package.train_model(model, X_train, y_train)
        state_before = (
            model.preprocessor.fit_calls,
            tuple(model.preprocessor.feature_columns),
            dict(model.preprocessor.numeric_fill_values),
            id(model.estimator.x_converter_),
        )
        predictions = np.asarray(package.predict_model(model, X_test)).reshape(-1)
        state_after = (
            model.preprocessor.fit_calls,
            tuple(model.preprocessor.feature_columns),
            dict(model.preprocessor.numeric_fill_values),
            id(model.estimator.x_converter_),
        )
        self.assertEqual(model.fit_calls, 1)
        self.assertEqual(state_before, state_after)
        self.assertEqual(predictions.shape, (len(y_test),))
        if task_type == "classification":
            probabilities = np.asarray(model.predict_proba(X_test))
            self.assertEqual(probabilities.shape, (len(y_test), 2))
            self.assertTrue(np.isfinite(probabilities).all())
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
            self.assertGreaterEqual(accuracy_score(y_test, predictions), 0.5)
        else:
            self.assertTrue(np.isfinite(predictions.astype(float)).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{package_name}-{task_type}.joblib"
            joblib.dump(model, path)
            loaded = joblib.load(path)
            loaded_predictions = np.asarray(loaded.predict(X_test)).reshape(-1)
        np.testing.assert_array_equal(loaded_predictions, predictions)
        self.assertEqual(loaded.preprocessor.fit_calls, 1)

    def test_realmlp_classification_and_regression_lifecycles(self):
        self._assert_lifecycle("realmlp", "classification")
        self._assert_lifecycle("realmlp", "regression")

    def test_tabm_classification_and_regression_lifecycles(self):
        self._assert_lifecycle("tabm", "classification")
        self._assert_lifecycle("tabm", "regression")

    def test_ft_transformer_classification_and_regression_lifecycles(self):
        self._assert_lifecycle("ft_transformer", "classification")
        self._assert_lifecycle("ft_transformer", "regression")

    def test_all_pytabkit_candidates_support_classification_and_regression(self):
        for package_name in ("realmlp", "tabm", "ft_transformer"):
            package = __import__(
                f"pipelines.pipeline2_prediction.models.{package_name}",
                fromlist=["get_model_info"],
            )
            self.assertEqual(
                package.get_model_info()["supported_tasks"],
                ["classification", "regression"],
            )
            self.assertIsNotNone(package.build_model("regression", device="cpu"))
            with self.assertRaisesRegex(ValueError, "Unsupported task"):
                package.build_model("clustering", device="cpu")


if __name__ == "__main__":
    unittest.main()
