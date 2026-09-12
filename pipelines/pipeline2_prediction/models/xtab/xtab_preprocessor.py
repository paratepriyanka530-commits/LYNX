import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


class XTabPreprocessor:
    """
    Generic tabular-data preprocessor for LYNX XTab.

    Supports:
        - Numerical features
        - Categorical features
        - Missing values
        - Numerical scaling
        - Categorical encoding
        - Classification
        - Regression

    The task can be explicitly specified or automatically detected.
    """

    def __init__(self, task=None):
        """
        Parameters
        ----------
        task : str or None
            "classification", "regression", or None.

            If None, the task is detected automatically.
        """

        if task not in {
            None,
            "classification",
            "regression"
        }:
            raise ValueError(
                "task must be 'classification', "
                "'regression', or None."
            )

        self.requested_task = task

        self.numerical_columns = []
        self.categorical_columns = []

        self.category_mappings = {}
        self.target_mapping = {}

        self.scaler = None
        self.target_scaler = None


        self.target_column = None
        self.task = None

    def _detect_task(self, target):
        """
        Automatically determine classification or regression.

        Automatic rule:
            - Non-numeric target -> classification
            - Numeric integer target with few unique values and a
              low unique-value ratio -> classification
            - Other numeric targets -> regression

        Explicit task selection is preferred when the target semantics
        are known.
        """

        if pd.api.types.is_numeric_dtype(target):

            unique_values = target.nunique()
            total_values = target.notna().sum()

            unique_ratio = (
                unique_values / total_values
                if total_values > 0
                else 0
            )

            if (
                pd.api.types.is_integer_dtype(target)
                and unique_values <= 10
                and unique_ratio <= 0.20
                
            ):
                return "classification"

            return "regression"

        return "classification"

    def fit(self, dataframe, target_column):
        """
        Learn preprocessing information from training data.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column '{target_column}' "
                f"does not exist in the dataset."
            )

        if dataframe.empty:
            raise ValueError(
                "Cannot fit preprocessor on an empty dataset."
            )

        self.target_column = target_column

        X = dataframe.drop(columns=[target_column])
        y = dataframe[target_column]

        # Determine task
        if self.requested_task is not None:
            self.task = self.requested_task
        else:
            self.task = self._detect_task(y)

        # Identify feature types
        self.numerical_columns = X.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        self.categorical_columns = X.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        # Numerical preprocessing
        if self.numerical_columns:

            numerical_data = X[
                self.numerical_columns
            ].copy()

            numerical_data = numerical_data.fillna(
                numerical_data.median()
            )

            self.scaler = StandardScaler()

            self.scaler.fit(numerical_data)

        else:
            self.scaler = None

        # Categorical preprocessing
        self.category_mappings = {}

        for column in self.categorical_columns:

            values = (
                X[column]
                .fillna("__MISSING__")
                .astype(str)
            )

            categories = sorted(
                values.unique().tolist()
            )

            # Reserve 0 for unknown categories
            mapping = {
                category: index + 1
                for index, category in enumerate(categories)
            }

            self.category_mappings[column] = mapping

        # Target preprocessing
        # Target preprocessing
        self.target_mapping = {}
        self.target_scaler = None

        if self.task == "classification":

            target_values = (
                y.fillna("__MISSING__")
                .astype(str)
            )

            classes = sorted(
               target_values.unique().tolist()
            )

            self.target_mapping = {
                value: index
                for index, value in enumerate(classes)
            }

        else:

            numeric_target = pd.to_numeric(
                y,
                errors="coerce"
        )

            if numeric_target.isna().any():
                raise ValueError(
                    "Regression target contains "
                    "missing or non-numeric values."
                )

            self.target_scaler = StandardScaler()

            self.target_scaler.fit(
                numeric_target.to_numpy(
                   dtype=np.float32
                ).reshape(-1, 1)
            )

    def transform_features(self, dataframe):
        """
        Transform feature columns into numerical
        and categorical arrays.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if self.target_column is None:
            raise ValueError(
                "Preprocessor has not been fitted yet."
            )

        if self.target_column in dataframe.columns:

            X = dataframe.drop(
                columns=[self.target_column]
            )

        else:

            X = dataframe.copy()

        # Check required feature columns
        required_columns = (
            self.numerical_columns
            + self.categorical_columns
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in X.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required feature columns: "
                f"{missing_columns}"
            )

        # Numerical features
        if self.numerical_columns:

            numerical_data = X[
                self.numerical_columns
            ].copy()

            numerical_data = numerical_data.fillna(
                pd.Series(
                    self.scaler.mean_,
                    index=self.numerical_columns
                )
            )

            numerical_data = self.scaler.transform(
                numerical_data
            )

            numerical_data = numerical_data.astype(
                np.float32
            )

        else:

            numerical_data = np.empty(
                (len(X), 0),
                dtype=np.float32
            )

        # Categorical features
        if self.categorical_columns:

            categorical_data = []

            for column in self.categorical_columns:

                mapping = self.category_mappings[column]

                values = (
                    X[column]
                    .fillna("__MISSING__")
                    .astype(str)
                )

                encoded = values.map(
                    lambda value: mapping.get(
                        value,
                        0
                    )
                )

                categorical_data.append(
                    encoded.to_numpy()
                )

            categorical_data = np.column_stack(
                categorical_data
            ).astype(np.int64)

        else:

            categorical_data = np.empty(
                (len(X), 0),
                dtype=np.int64
            )

        return numerical_data, categorical_data

    def transform_target(self, target):
        """
        Transform target values.
        """

        if self.task is None:
            raise ValueError(
                "Preprocessor has not been fitted yet."
            )

        if self.task == "classification":

            values = (
                target.fillna("__MISSING__")
                .astype(str)
            )

            unknown_values = (
                set(values)
                - set(self.target_mapping.keys())
            )

            if unknown_values:
                raise ValueError(
                    "Unknown target values found during "
                    f"transformation: {unknown_values}"
                )

            encoded = values.map(
                self.target_mapping
            )

            return encoded.to_numpy(
                dtype=np.int64
            )

        numeric_target = pd.to_numeric(
            target,
            errors="coerce"
        )

        if numeric_target.isna().any():
            raise ValueError(
                "Regression target contains "
                "missing or non-numeric values."
            )

        scaled_target = self.target_scaler.transform(
              numeric_target.to_numpy(
                 dtype=np.float32
              ).reshape(-1, 1)
            )

        return scaled_target.squeeze(1).astype(
            np.float32
        )
    def inverse_transform_target(self, target):
        """
        Convert transformed regression targets back
        to their original scale.
         """

        if self.task != "regression":
           raise ValueError(
               "inverse_transform_target is only available "
                "for regression."
            )

        if self.target_scaler is None:
            raise ValueError(
                "Target scaler has not been fitted."
            )

        target = np.asarray(
          target,
          dtype=np.float32
    )

        return self.target_scaler.inverse_transform(
           target.reshape(-1, 1)
       ).squeeze(1)
    def fit_transform(
        self,
        dataframe,
        target_column
    ):
        """
        Fit preprocessing information and transform
        the dataset.
        """

        self.fit(
            dataframe,
            target_column
        )

        numerical_data, categorical_data = (
            self.transform_features(dataframe)
        )

        target = dataframe[target_column]

        target_data = self.transform_target(
            target
        )

        return (
            numerical_data,
            categorical_data,
            target_data
        )

    @property
    def num_classes(self):
        if self.task != "classification":
            return None

        return len(self.target_mapping)

    @property
    def categorical_cardinalities(self):
        return [
            len(mapping) + 1
            for mapping in self.category_mappings.values()
        ]

