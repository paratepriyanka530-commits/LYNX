import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pipelines.pipeline2_prediction.models.catboost.catboost_predictor import (
    CatBoostPredictor,
)
from pipelines.pipeline2_prediction.models.catboost.catboost_model import (
    LYNXCatBoost,
)
from pipelines.pipeline2_prediction.models.catboost.catboost_preprocessor import (
    CatBoostPreprocessor,
)
from pipelines.pipeline2_prediction.models.lightgbm.lightgbm_predictor import (
    LightGBMPredictor,
)
from pipelines.pipeline2_prediction.models.lightgbm.lightgbm_model import (
    LYNXLightGBM,
)
from pipelines.pipeline2_prediction.models.lightgbm.lightgbm_preprocessor import (
    LightGBMPreprocessor,
)
from pipelines.pipeline2_prediction.models.xgboost.xgboost_model import LYNXXGBoost
from pipelines.pipeline2_prediction.models.xgboost.xgboost_preprocessor import (
    XGBoostPreprocessor,
)


BLOOD_PATH = "benchmarks/tabarena/datasets/blood-transfusion-service-center.csv"
SECOND_DATASET_PATH = "benchmarks/tabarena/datasets/diabetes.csv"
SECOND_TARGET = "TestedPositiveForDiabetes"


class SavedArtifactSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blood = pd.read_csv(BLOOD_PATH)
        diabetes = pd.read_csv(SECOND_DATASET_PATH)
        cls.diabetes_features = diabetes.drop(columns=[SECOND_TARGET])
        cls.diabetes_target = diabetes[SECOND_TARGET]
        (
            cls.diabetes_train,
            cls.diabetes_test,
            cls.target_train,
            cls.target_test,
        ) = train_test_split(
            cls.diabetes_features,
            cls.diabetes_target,
            test_size=0.20,
            random_state=42,
            stratify=cls.diabetes_target,
        )

    def _assert_compatible_prediction(self, predictor, raw_model):
        schema = predictor.preprocessor.feature_columns
        preprocessor_id = id(predictor.preprocessor)
        transformed = predictor.preprocessor.transform(self.blood[schema].tail(7))
        predictions = np.asarray(raw_model.predict(transformed)).reshape(-1)
        probabilities = np.asarray(raw_model.predict_proba(transformed))
        self.assertEqual(predictions.shape, (7,))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertEqual(probabilities.shape, (7, 2))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(7))
        self.assertEqual(preprocessor_id, id(predictor.preprocessor))

    def test_lightgbm_artifact_is_healthy_but_second_schema_is_rejected(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predictor = LightGBMPredictor()
            self._assert_compatible_prediction(predictor, predictor.model.model)
        # Warning-free behavior is asserted in the pinned artifact environment;
        # this suite also runs under developers' potentially newer sklearn.
        with self.assertRaisesRegex(ValueError, "missing columns"):
            predictor.preprocessor.transform(self.diabetes_features)

    def test_catboost_artifact_is_healthy_but_second_schema_is_rejected(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predictor = CatBoostPredictor()
            self._assert_compatible_prediction(predictor, predictor.model)
        # Warning-free behavior is asserted in the pinned artifact environment.
        with self.assertRaisesRegex(ValueError, "missing columns"):
            predictor.preprocessor.transform(self.diabetes_features)

    def _assert_fresh_lifecycle(self, preprocessor_class, model_class):
        self.assertNotIn(SECOND_TARGET, self.diabetes_features.columns)
        self.assertTrue(
            set(self.diabetes_train.index).isdisjoint(self.diabetes_test.index)
        )

        target_encoder = LabelEncoder()
        y_train = target_encoder.fit_transform(self.target_train.astype(str))
        y_test = target_encoder.transform(self.target_test.astype(str))

        preprocessor = preprocessor_class()
        transformed_train = preprocessor.fit_transform(self.diabetes_train)
        state_before_test = (
            list(preprocessor.feature_columns),
            dict(preprocessor.numeric_fill_values),
            tuple(getattr(preprocessor, "categorical_columns", ())),
            {
                column: tuple(encoder.classes_)
                for column, encoder in getattr(preprocessor, "encoders", {}).items()
            },
        )
        transformed_test = preprocessor.transform(self.diabetes_test)
        state_after_test = (
            list(preprocessor.feature_columns),
            dict(preprocessor.numeric_fill_values),
            tuple(getattr(preprocessor, "categorical_columns", ())),
            {
                column: tuple(encoder.classes_)
                for column, encoder in getattr(preprocessor, "encoders", {}).items()
            },
        )

        self.assertEqual(state_before_test, state_after_test)
        self.assertEqual(
            list(transformed_train.columns),
            list(transformed_test.columns),
        )
        for column, fill_value in preprocessor.numeric_fill_values.items():
            self.assertEqual(fill_value, self.diabetes_train[column].median())

        model = model_class()
        another_fresh_model = model_class()
        self.assertIsNot(model, another_fresh_model)
        self.assertFalse(model.is_trained)
        model.train(transformed_train, y_train)
        predictions = np.asarray(model.predict(transformed_test)).reshape(-1)
        probabilities = np.asarray(model.model.predict_proba(transformed_test))

        self.assertEqual(predictions.shape, (len(y_test),))
        self.assertGreater(np.unique(predictions).size, 1)
        self.assertTrue(np.isfinite(predictions).all())
        self.assertEqual(probabilities.shape, (len(y_test), 2))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(len(y_test)))

    def test_lightgbm_fresh_diabetes_lifecycle(self):
        self._assert_fresh_lifecycle(LightGBMPreprocessor, LYNXLightGBM)

    def test_catboost_fresh_diabetes_lifecycle(self):
        self._assert_fresh_lifecycle(CatBoostPreprocessor, LYNXCatBoost)

    def test_xgboost_fresh_diabetes_lifecycle(self):
        self._assert_fresh_lifecycle(XGBoostPreprocessor, LYNXXGBoost)


if __name__ == "__main__":
    unittest.main()
