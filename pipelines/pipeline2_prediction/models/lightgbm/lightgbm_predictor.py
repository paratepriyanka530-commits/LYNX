"""
LYNX LightGBM Predictor

Responsibilities:
    - Load prediction dataset
    - Preprocess dataset
    - Load trained model
    - Generate predictions
    - Save predictions
"""

from pathlib import Path

import pandas as pd

from ..artifact_io import load_joblib_artifact, reference_artifact_path
from .lightgbm_preprocessor import LightGBMPreprocessor

class LightGBMPredictor:

    def __init__(
        self,
        model_path: str = str(
            reference_artifact_path("lightgbm", "lightgbm_model.pkl")
        ),
        preprocessor_path: str = str(
            reference_artifact_path("lightgbm", "lightgbm_preprocessor.pkl")
        ),
    ) -> None:

        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

        self.model = self._load_model()
        self.preprocessor: LightGBMPreprocessor = self._load_preprocessor()

    def _load_model(self):

        model_file = Path(
            self.model_path
        )

        if not model_file.exists():

            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        return load_joblib_artifact(
            model_file,
            legacy_modules=(
                "models.lightgbm",
                "models.lightgbm.lightgbm_model",
            ),
        )

    def _load_preprocessor(self):

        preprocessor_file = Path(self.preprocessor_path)

        if not preprocessor_file.exists():
            raise FileNotFoundError(
                "Fitted LightGBM preprocessor not found: "
                f"{self.preprocessor_path}. Retrain the model to create it."
            )

        return load_joblib_artifact(
            preprocessor_file,
            legacy_modules=(
                "models.lightgbm",
                "models.lightgbm.lightgbm_preprocessor",
            ),
        )

    def predict(
        self,
        csv_path: str,
        output_path: str = "predictions.csv",
    ) -> pd.DataFrame:

        print("Loading dataset...")

        data = pd.read_csv(
            csv_path
        )

        print(
            f"Dataset Shape: {data.shape}"
        )

        print(
            "Preprocessing dataset..."
        )

        X = self.preprocessor.transform(
            data
        )

        print(
            "Generating predictions..."
        )

        predictions = self.model.predict(
            X
        )

        result = data.copy()

        result["prediction"] = predictions

        result.to_csv(
            output_path,
            index=False,
        )

        print(
            f"Predictions saved to: {output_path}"
        )

        return result


if __name__ == "__main__":

    model_path = input(
        "Enter model path: "
    ).strip().replace('"', "")

    csv_path = input(
        "Enter CSV path: "
    ).strip().replace('"', "")

    predictor = LightGBMPredictor(
        model_path,
        input("Enter fitted preprocessor path: ").strip().replace('"', ""),
    )

    predictor.predict(
        csv_path
    )

    print(
        "\nPrediction completed successfully."
    )
