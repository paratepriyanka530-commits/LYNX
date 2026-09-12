"""
LYNX XGBoost Predictor

Responsibilities:
    - Load trained model
    - Load preprocessor
    - Load prediction dataset
    - Generate predictions
    - Save predictions
"""

from pathlib import Path

import pandas as pd

from ..artifact_io import load_joblib_artifact, reference_artifact_path

class XGBoostPredictor:

    def __init__(
        self,
        model_path: str = str(
            reference_artifact_path("xgboost", "xgboost_model.pkl")
        ),
        preprocessor_path: str = str(
            reference_artifact_path("xgboost", "xgboost_preprocessor.pkl")
        ),
    ):

        self.model = self._load_model(
            model_path
        )

        self.preprocessor = self._load_preprocessor(
            preprocessor_path
        )

    def _load_model(
        self,
        model_path: str,
    ):

        path = Path(model_path)

        if not path.exists():

            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        return load_joblib_artifact(model_path)

    def _load_preprocessor(
        self,
        preprocessor_path: str,
    ):

        path = Path(preprocessor_path)

        if not path.exists():

            raise FileNotFoundError(
                f"Preprocessor file not found: {preprocessor_path}"
            )

        return load_joblib_artifact(
            preprocessor_path,
            legacy_modules=(
                "models.xgboost",
                "models.xgboost.xgboost_preprocessor",
            ),
        )

    def predict(
        self,
        csv_path: str,
        output_path: str = "predictions.csv",
    ):

        print(
            "Loading prediction dataset..."
        )

        data = pd.read_csv(
            csv_path
        )

        print(
            f"Dataset Shape: {data.shape}"
        )

        print(
            "Applying preprocessing..."
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

        results = data.copy()

        results["prediction"] = predictions

        results.to_csv(
            output_path,
            index=False,
        )

        print(
            f"Predictions saved to: {output_path}"
        )

        return results


if __name__ == "__main__":

    csv_path = input(
        "Enter CSV file path: "
    ).strip().replace('"', "")

    predictor = XGBoostPredictor()

    predictor.predict(
        csv_path
    )

    print(
        "\nPrediction completed successfully."
    )
