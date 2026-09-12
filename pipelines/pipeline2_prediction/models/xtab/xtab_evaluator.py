import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class XTabEvaluator:
    """
    Evaluation module for the LYNX XTab model.

    Supports:
        - Classification
        - Regression
    """

    @staticmethod
    def classification_metrics(y_true, y_pred):
        """
        Calculate classification metrics.
        """

        report = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        return {
            "accuracy": accuracy_score(
                y_true,
                y_pred
            ),
            "precision": precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            "f1": f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred
            ).tolist(),
            "per_class": report
        }

    @staticmethod
    def regression_metrics(y_true, y_pred):
        """
        Calculate regression metrics.
        """

        mse = mean_squared_error(
            y_true,
            y_pred
        )

        rmse = np.sqrt(mse)

        return {
            "mae": mean_absolute_error(
                y_true,
                y_pred
            ),
            "mse": mse,
            "rmse": rmse,
            "r2": r2_score(
                y_true,
                y_pred
            )
        }

    def evaluate(
        self,
        y_true,
        y_pred,
        task
    ):
        """
        Automatically select the correct
        evaluation metrics.
        """

        if task == "classification":

            return self.classification_metrics(
                y_true,
                y_pred
            )

        if task == "regression":

            return self.regression_metrics(
                y_true,
                y_pred
            )

        raise ValueError(
            "task must be 'classification' or 'regression'."
        )

