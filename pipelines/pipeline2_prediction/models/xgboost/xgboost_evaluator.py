"""
LYNX XGBoost Evaluator

Responsibilities:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class XGBoostEvaluator:

    @staticmethod
    def evaluate(
        y_true,
        y_pred,
    ):

        results = {

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

        return results

    @staticmethod
    def print_results(
        results,
    ):

        print("\n===== LYNX XGBoost Evaluation =====")

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
