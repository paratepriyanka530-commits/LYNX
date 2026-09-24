"""
LYNX CatBoost Model
"""

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from .catboost_config import CATBOOST_CONFIG


class LYNXCatBoost:
    """
    Task-aware CatBoost model for LYNX.
    """

    def __init__(self, task_type: str = "classification") -> None:

        if task_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported CatBoost task type: {task_type}")
        self.task_type = task_type
        config = dict(CATBOOST_CONFIG)
        if task_type == "regression":
            config["loss_function"] = "RMSE"
            self.model = CatBoostRegressor(**config)
        else:
            self.model = CatBoostClassifier(**config)

        self.is_trained = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train,
    ) -> "LYNXCatBoost":

        self._validate_training_data(
            X_train,
            y_train,
        )

        # Automatically detect categorical columns
        categorical_features = list(
            X_train.select_dtypes(
                include=["object", "string"]
            ).columns
        )

        if self.task_type == "classification" and pd.Series(y_train).nunique() > 2:
            self.model.set_params(loss_function="MultiClass")

        self.model.fit(
            X_train,
            y_train,
            cat_features=categorical_features,
        )

        self.is_trained = True

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ):

        self._check_trained()

        return self.model.predict(X)

    def predict_probability(
        self,
        X: pd.DataFrame,
    ):

        self._check_trained()

        if self.task_type != "classification":
            raise RuntimeError("Regression models do not expose class probabilities.")
        return self.model.predict_proba(X)

    def get_feature_importance(
        self,
        X: pd.DataFrame,
        y,
    ) -> pd.DataFrame:

        self._check_trained()

        importance = pd.DataFrame(
            {
                "feature": X.columns,
                "importance": self.model.get_feature_importance(),
            }
        )

        return importance.sort_values(
            by="importance",
            ascending=False,
        )

    def _validate_training_data(
        self,
        X_train,
        y_train,
    ) -> None:

        if len(X_train) == 0:
            raise ValueError(
                "Training dataset is empty."
            )

        if len(y_train) == 0:
            raise ValueError(
                "Target column is empty."
            )

    def _check_trained(
        self,
    ) -> None:

        if not self.is_trained:
            raise RuntimeError(
                "CatBoost model has not been trained yet."
            )
