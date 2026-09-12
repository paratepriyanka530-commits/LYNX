"""
LYNX - XGBoost Model

Input:
    CSV dataset

The CSV must contain:
    - Feature columns
    - One target/label column

Example:
    python xgboost_model.py data.csv target
"""

import sys
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from .xgboost_config import XGBOOST_CONFIG


class LYNXXGBoost:
    """
    XGBoost binary classification model for LYNX.
    """

    def __init__(self) -> None:

        self.model = XGBClassifier(
            **XGBOOST_CONFIG
        )

        self.is_trained = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> "LYNXXGBoost":

        self._validate_training_data(
            X_train,
            y_train,
        )

        self.model.fit(
            X_train,
            y_train,
        )

        self.is_trained = True

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:

        self._check_trained()

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

        probabilities = self.model.predict_proba(X)[:, 1]

        return pd.Series(
            probabilities,
            index=X.index,
            name="prediction_probability",
        )

    def get_feature_importance(
        self,
    ) -> pd.DataFrame:

        self._check_trained()

        importance = pd.DataFrame({
            "feature": self.model.feature_names_in_,
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

        if len(X_train) != len(y_train):
            raise ValueError(
                "Features and labels must contain the same number of rows."
            )

        if pd.Series(y_train).nunique() < 2:
            raise ValueError(
                "Target column must contain at least two classes."
            )

    def _check_trained(
        self,
    ) -> None:

        if not self.is_trained:
            raise RuntimeError(
                "The LYNX XGBoost model has not been trained yet."
            )


def load_csv(
    csv_path: str,
) -> pd.DataFrame:

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
            f"Target column '{target_column}' was not found."
        )

    X = data.drop(
        columns=[target_column]
    )

    y = data[target_column]

    return X, y


def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "python xgboost_model.py <csv_path> <target_column>"
        )

        return

    csv_path = sys.argv[1]
    target_column = sys.argv[2]

    data = load_csv(
        csv_path
    )

    X, y = prepare_data(
        data,
        target_column,
    )

    model = LYNXXGBoost()

    model.train(
        X,
        y,
    )

    print(
        "LYNX XGBoost model trained successfully."
    )


if __name__ == "__main__":
    main()
