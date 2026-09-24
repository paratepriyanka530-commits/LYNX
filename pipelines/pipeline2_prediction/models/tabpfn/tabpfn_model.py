"""
LYNX - TabPFN Model

Uses the official pretrained TabPFN classifier.

TabPFN is a pretrained foundation model for tabular data.
The model itself is not modified or retrained from scratch.
"""

from typing import Optional
import pickle

import pandas as pd
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import (
    TabPFNHuggingFaceGatedRepoError,
    TabPFNLicenseError,
)

from ..artifact_io import load_pickle_artifact


class TabPFNProvisioningError(RuntimeError):
    """Raised when official TabPFN weights cannot be provisioned."""


_PROVISIONING_ERROR = (
    "TabPFN model weights are unavailable. TabPFN requires one-time "
    "PriorLabs license acceptance at https://ux.priorlabs.ai. Interactive "
    "browser login may be used when allowed. For non-interactive/headless "
    "execution, set TABPFN_TOKEN to a valid PriorLabs API key before "
    "training. The official model_path='auto' cache/download path was used; "
    "no reference pickle fallback was attempted."
)


def _is_provisioning_error(error: BaseException) -> bool:
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        if isinstance(
            current,
            (TabPFNLicenseError, TabPFNHuggingFaceGatedRepoError),
        ):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


class LYNXTabPFN:
    """
    LYNX wrapper around the official task-appropriate TabPFN estimator.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_path=None,
        task_type: str = "classification",
    ) -> None:

        if task_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported TabPFN task type: {task_type}")
        self.task_type = task_type
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if model_path is not None:
            kwargs["model_path"] = model_path
        estimator = TabPFNClassifier if task_type == "classification" else TabPFNRegressor
        self.model = estimator(**kwargs)

        self.is_trained = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> "LYNXTabPFN":

        self._validate_training_data(
            X_train,
            y_train,
        )

        try:
            self.model.fit(
                X_train,
                y_train,
            )
        except (TabPFNLicenseError, TabPFNHuggingFaceGatedRepoError) as error:
            raise TabPFNProvisioningError(_PROVISIONING_ERROR) from error
        except RuntimeError as error:
            if _is_provisioning_error(error):
                raise TabPFNProvisioningError(_PROVISIONING_ERROR) from error
            raise

        self.is_trained = True

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ):

        self._check_trained()

        return self.model.predict(
            X
        )

    def predict_probability(
        self,
        X: pd.DataFrame,
    ):

        self._check_trained()

        if self.task_type != "classification":
            raise RuntimeError("Regression models do not expose class probabilities.")
        return self.model.predict_proba(
            X
        )

    def save_model(
        self,
        filepath: str,
    ) -> None:
        """
        Save fitted TabPFN model.
        """

        self._check_trained()

        with open(filepath, "wb") as file:
            pickle.dump(
                self.model,
                file
            )

    def load_model(
        self,
        filepath: str,
    ) -> None:
        """
        Load saved TabPFN model.
        """

        self.model = load_pickle_artifact(filepath)

        self.is_trained = True

    def _validate_training_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:

        if not isinstance(
            X_train,
            pd.DataFrame
        ):
            raise TypeError(
                "X_train must be a pandas DataFrame."
            )

        if X_train.empty:
            raise ValueError(
                "X_train cannot be empty."
            )

        if len(X_train) != len(y_train):
            raise ValueError(
                "X_train and y_train must have the same number of rows."
            )

        if self.task_type == "classification" and pd.Series(y_train).nunique() < 2:
            raise ValueError(
                "y_train must contain at least two classes."
            )

    def _check_trained(
        self,
    ) -> None:

        if not self.is_trained:
            raise RuntimeError(
                "The LYNX TabPFN model has not been fitted yet."
            )
