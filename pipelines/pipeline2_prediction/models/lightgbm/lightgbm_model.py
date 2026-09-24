"""
LYNX - LightGBM Model

Input:
    CSV dataset

The CSV must contain:
    - Feature columns
    - One target/label column

Example:
    python lightgbm_model.py data.csv target
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .cb_chl import make_cb_chl_objective
from .lightgbm_config import LIGHTGBM_CONFIG


class LYNXLightGBM:
    """
    Task-aware LightGBM model for LYNX.
    """

    def __init__(
        self,
        training_mode: str = "standard",
        cb_chl_eligible: bool = False,
        cost_config=None,
        task_type: str = "classification",
    ) -> None:

        if training_mode not in {"standard", "cb_chl"}:
            raise ValueError(f"Unsupported LightGBM training_mode: {training_mode}")

        if task_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported LightGBM task type: {task_type}")
        if task_type == "regression" and training_mode != "standard":
            raise ValueError("CB-CHL is classification-only.")
        self.task_type = task_type
        config = dict(LIGHTGBM_CONFIG)
        if task_type == "regression":
            config.update(objective="regression", metric="rmse")
            self.model = lgb.LGBMRegressor(**config)
        else:
            self.model = lgb.LGBMClassifier(**config)

        self.training_mode = training_mode
        self.cb_chl_eligible = bool(cb_chl_eligible)
        self.cost_config = cost_config
        self.cb_chl_training_state = None
        self.is_trained = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> "LYNXLightGBM":

        self._validate_training_data(X_train, y_train)

        if self.task_type == "classification" and self.training_mode == "standard":
            n_classes = int(pd.Series(y_train).nunique())
            if n_classes > 2:
                self.model.set_params(objective="multiclass", num_class=n_classes)

        if self.training_mode == "cb_chl":
            objective, state = make_cb_chl_objective(y_train, self.cost_config)
            self.cb_chl_training_state = state
            self.model.set_params(objective=objective)

        self.model.fit(X_train, y_train)

        self.is_trained = True

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:

        self._check_trained()

        if self.training_mode == "cb_chl":
            raw_scores = np.asarray(
                self.model.booster_.predict(X, raw_score=True), dtype=float
            )
            predictions = (raw_scores >= 0.0).astype(int)
        else:
            predictions = self.model.predict(X)

        return pd.Series(
            predictions,
            index=X.index,
            name="prediction",
        )

    def predict_probability(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:

        self._check_trained()

        if self.task_type != "classification":
            raise RuntimeError("Regression models do not expose class probabilities.")
        if self.training_mode == "cb_chl":
            raw_scores = np.asarray(
                self.model.booster_.predict(X, raw_score=True), dtype=float
            )
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -709.0, 709.0)))
        else:
            return self.model.predict_proba(X)

        return pd.Series(probabilities, index=X.index, name="prediction_probability")

    def get_feature_importance(self) -> pd.DataFrame:

        self._check_trained()

        importance = pd.DataFrame({
            "feature": self.model.feature_name_,
            "importance": self.model.feature_importances_,
        })

        return importance.sort_values(
            by="importance",
            ascending=False,
        ).reset_index(drop=True)

    def _validate_training_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:

        if X_train.empty:
            raise ValueError(
                "Training features cannot be empty."
            )

        if len(y_train) == 0:
            raise ValueError(
                "Training labels cannot be empty."
            )

        if len(X_train) != len(y_train):
            raise ValueError(
                "Features and labels must contain the same number of rows."
            )

        if self.task_type == "classification" and pd.Series(y_train).nunique() < 2:
            raise ValueError(
                "Target column must contain at least two classes."
            )

    def _check_trained(self) -> None:

        if not self.is_trained:
            raise RuntimeError(
                "The LYNX LightGBM model has not been trained yet."
            )


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset.
    """

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            "Input file must be a CSV file."
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError(
            "The CSV file is empty."
        )

    return data


def prepare_data(
    data: pd.DataFrame,
    target_column: str,
):

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the CSV."
        )

    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X, y


def main():

    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "python lightgbm_model.py <csv_path> <target_column>"
        )
        return

    csv_path = sys.argv[1]
    target_column = sys.argv[2]

    print("Loading CSV...")

    data = load_csv(csv_path)

    print(f"Dataset loaded successfully.")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    X, y = prepare_data(
        data,
        target_column,
    )

    print(f"Features: {X.shape[1]}")
    print(f"Target: {target_column}")

    print("Training LightGBM...")

    model = LYNXLightGBM()

    model.train(
        X,
        y,
    )

    print("LYNX LightGBM model trained successfully.")


if __name__ == "__main__":
    main()
