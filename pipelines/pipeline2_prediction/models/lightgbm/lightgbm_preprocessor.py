"""
LYNX LightGBM Preprocessor

Responsibilities:
    - Handle missing values
    - Encode categorical columns
    - Return clean dataset
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class LightGBMPreprocessor:

    def __init__(self):
        self.encoders = {}
        self.numeric_fill_values = {}
        self.feature_columns = []
        self.is_fitted = False

    def fit_transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        data = data.copy()
        self.feature_columns = list(data.columns)

        # Handle missing values
        for column in data.columns:

            if pd.api.types.is_numeric_dtype(
                data[column]
            ):

                fill_value = data[column].median()
                self.numeric_fill_values[column] = fill_value
                data[column] = data[column].fillna(fill_value)

            else:

                data[column] = data[column].fillna(
                    "Unknown"
                )

        # Find categorical columns
        categorical_columns = data.select_dtypes(
            include=["object", "string", "category"]
        ).columns

        # Encode categorical columns
        for column in categorical_columns:

            encoder = LabelEncoder()

            data[column] = encoder.fit_transform(
                data[column].astype(str)
            )

            self.encoders[column] = encoder

        self.is_fitted = True
        return data

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        if not getattr(self, "is_fitted", False):
            raise RuntimeError(
                "LightGBM preprocessor is not fitted. Retrain the model "
                "to create a fitted preprocessor artifact."
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

        for column, encoder in self.encoders.items():

            if column in data.columns:

                data[column] = encoder.transform(
                    data[column].fillna("Unknown").astype(str)
                )

        return data
