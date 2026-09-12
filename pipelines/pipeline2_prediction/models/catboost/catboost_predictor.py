"""
LYNX CatBoost Predictor

Responsibilities:
    - Load trained CatBoost model
    - Load preprocessor
    - Load prediction dataset
    - Apply preprocessing
    - Generate predictions
"""

from pathlib import Path

from ..artifact_io import load_joblib_artifact, reference_artifact_path
from .data_loader import load_csv


class CatBoostPredictor:
    """
    Handles CatBoost predictions.
    """

    def __init__(
        self,
        model_path: str = str(
            reference_artifact_path("catboost", "catboost_model.pkl")
        ),
        preprocessor_path: str = str(
            reference_artifact_path("catboost", "catboost_preprocessor.pkl")
        ),
    ) -> None:

        print("Loading trained model...")

        self.model = load_joblib_artifact(model_path)

        self.preprocessor = load_joblib_artifact(
            preprocessor_path,
            legacy_modules=(
                "models.catboost",
                "models.catboost.catboost_preprocessor",
            ),
        )

    def predict(
        self,
        csv_path: str,
    ):

        print(
            "Loading prediction dataset..."
        )

        data = load_csv(
            csv_path
        )

        print(
            f"Dataset Shape: {data.shape}"
        )

        print(
            "Applying preprocessing..."
        )

        data = self.preprocessor.transform(
            data
        )

        print(
            "Generating predictions..."
        )

        predictions = self.model.predict(
            data
        )

        print(
            f"Total Predictions: {len(predictions)}"
        )

        print(
            "\nSample Predictions:"
        )

        print(
            predictions[:10]
        )

        return predictions


if __name__ == "__main__":

    csv_path = input(
        "Enter CSV file path: "
    )

    predictor = CatBoostPredictor()

    predictor.predict(
        csv_path
    )

    print(
        "\nPrediction completed successfully."
    )
    
