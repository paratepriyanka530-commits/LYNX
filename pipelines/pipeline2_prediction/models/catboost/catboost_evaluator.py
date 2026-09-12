"""
LYNX CatBoost Evaluator
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class CatBoostEvaluator:
    """
    Evaluates CatBoost model performance.
    """

    @staticmethod
    def evaluate(
        y_true,
        y_pred,
    ):

        return {
            "accuracy": accuracy_score(
                y_true,
                y_pred,
            ),
            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
            ),
        }

    @staticmethod
    def print_results(
        results,
    ) -> None:

        print(
            "\n===== LYNX CatBoost Evaluation ====="
        )

        print(
            f"Accuracy  : {results['accuracy']:.4f}"
        )

        print(
            f"Precision : {results['precision']:.4f}"
        )

        print(
            f"Recall    : {results['recall']:.4f}"
        )

        print(
            f"F1 Score  : {results['f1_score']:.4f}"
        )

        print(
            "\nConfusion Matrix:"
        )

        print(
            results["confusion_matrix"]
        )
        
