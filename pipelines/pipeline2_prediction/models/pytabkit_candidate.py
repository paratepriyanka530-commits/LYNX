"""Shared leakage-safe adapter for pytabkit neural tabular estimators."""

from __future__ import annotations

import pandas as pd


class PytabkitPreprocessor:
    """Persist only train-derived null handling and input schema."""

    def __init__(self):
        self.feature_columns = []
        self.numeric_fill_values = {}
        self.categorical_columns = []
        self.is_fitted = False
        self.fit_calls = 0

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.fit_calls += 1
        if self.fit_calls != 1:
            raise RuntimeError("Pytabkit preprocessing may only be fitted once")
        result = frame.copy()
        self.feature_columns = list(result.columns)
        self.categorical_columns = list(
            result.select_dtypes(include=["object", "string", "category", "bool"]).columns
        )
        for column in self.feature_columns:
            if column in self.categorical_columns:
                result[column] = result[column].astype("object").fillna("__LYNX_MISSING__")
            else:
                fill_value = result[column].median()
                if pd.isna(fill_value):
                    fill_value = 0.0
                self.numeric_fill_values[column] = fill_value
                result[column] = result[column].fillna(fill_value)
        self.is_fitted = True
        return result

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Pytabkit preprocessor must be fitted before transform")
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Prediction data is missing columns: {missing}")
        result = frame.loc[:, self.feature_columns].copy()
        for column, fill_value in self.numeric_fill_values.items():
            result[column] = result[column].fillna(fill_value)
        for column in self.categorical_columns:
            result[column] = result[column].astype("object").fillna("__LYNX_MISSING__")
        return result


class PytabkitCandidate:
    """Own the fitted preprocessor and pytabkit estimator as one candidate."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.preprocessor = PytabkitPreprocessor()
        self.fit_calls = 0

    def fit(self, X_train, y_train):
        self.fit_calls += 1
        if self.fit_calls != 1:
            raise RuntimeError("Candidate models must be freshly built for each dataset")
        transformed = self.preprocessor.fit_transform(X_train)
        self.estimator.fit(transformed, y_train)
        return self

    def predict(self, X_test):
        return self.estimator.predict(self.preprocessor.transform(X_test))

    def predict_proba(self, X_test):
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError("Regression estimators do not expose predict_proba")
        return self.estimator.predict_proba(self.preprocessor.transform(X_test))
