"""
LYNX CatBoost Preprocessor
"""

import pandas as pd


class CatBoostPreprocessor:
    """
    Handles dataset preprocessing for CatBoost.
    """

    def __init__(self) -> None:
        self.numeric_fill_values = {}
        self.categorical_columns = []
        self.feature_columns = []
        self.is_fitted = False

    def fit_transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        data = data.copy()
        self.feature_columns = list(data.columns)

        # Fill missing numeric values
        numeric_columns = data.select_dtypes(
            include=["int64", "float64"]
        ).columns

        for column in numeric_columns:
            fill_value = data[column].median()
            self.numeric_fill_values[column] = fill_value
            data[column] = data[column].fillna(fill_value)

        # Fill missing categorical values
        categorical_columns = data.select_dtypes(
            include=["object", "string"]
        ).columns
        self.categorical_columns = list(categorical_columns)

        for column in categorical_columns:
            data[column] = data[column].fillna(
                "Unknown"
            )

        self.is_fitted = True
        return data

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        if not getattr(self, "is_fitted", False):
            raise RuntimeError(
                "CatBoost preprocessor artifact has no fitted training state. "
                "Retrain the model to create a usable preprocessor artifact."
            )

        data = data.copy()

        missing_columns = set(self.feature_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"Prediction data is missing columns: {sorted(missing_columns)}"
            )

        data = data[self.feature_columns]

        for column, fill_value in self.numeric_fill_values.items():
            data[column] = data[column].fillna(fill_value)

        for column in self.categorical_columns:
            data[column] = data[column].fillna("Unknown")

        return data
