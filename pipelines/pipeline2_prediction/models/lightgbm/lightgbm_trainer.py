"""
LYNX LightGBM Trainer

Responsibilities:
    - Load dataset
    - Prepare features and target
    - Preprocess data
    - Train/Test split
    - Train LightGBM model
    - Generate predictions
    - Evaluate model performance
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

from .lightgbm_model import (
    LYNXLightGBM,
    load_csv,
    prepare_data,
)

from .lightgbm_evaluator import LightGBMEvaluator
from .lightgbm_preprocessor import LightGBMPreprocessor
from ..artifact_io import training_artifact_dir


class LightGBMTrainer:
    """
    Handles LightGBM training workflow.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:

        self.test_size = test_size
        self.random_state = random_state

    def train(
        self,
        csv_path: str,
        target_column: str,
        dataset_id: str | None = None,
        run_id: str | None = None,
        artifact_root: str | Path | None = None,
    ):
        """
        Train LightGBM using CSV input.
        """

        print("Loading dataset...")

        data = load_csv(csv_path)

        print(
            f"Dataset Shape: {data.shape}"
        )

        # --------------------------------------------------
        # Prepare Features and Target
        # --------------------------------------------------

        X, y = prepare_data(
            data,
            target_column,
        )

        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------

        print("Preprocessing dataset...")

        if not pd.api.types.is_numeric_dtype(y):

            target_encoder = LabelEncoder()

            y = target_encoder.fit_transform(
                y.astype(str)
            )

        # --------------------------------------------------
        # Train/Test Split
        # --------------------------------------------------

        print(
            "Performing train/test split..."
        )

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        preprocessor = LightGBMPreprocessor()
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        print(
            f"Training Rows: {len(X_train)}"
        )

        print(
            f"Testing Rows: {len(X_test)}"
        )

        # --------------------------------------------------
        # Model Training
        # --------------------------------------------------

        model = LYNXLightGBM()

        print("Training LightGBM...")

        model.train(
            X_train,
            y_train,
        )

        artifact_dir = training_artifact_dir(
            "lightgbm", dataset_id or Path(csv_path).stem, run_id, artifact_root
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, artifact_dir / "lightgbm_model.pkl")

        joblib.dump(preprocessor, artifact_dir / "lightgbm_preprocessor.pkl")

        print(
            "LightGBM model and fitted preprocessor saved successfully."
        )

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        predictions = model.predict(
            X_test
        )

        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------

        results = LightGBMEvaluator.evaluate(
            y_test,
            predictions
        )

        LightGBMEvaluator.print_results(
            results
        )

        print(
            "LightGBM training completed successfully."
        )

        return {
            "model": model,
            "preprocessor": preprocessor,
            "artifact_dir": artifact_dir,
            "X_test": X_test,
            "X_test_raw": X_test_raw,
            "y_test": y_test,
            "predictions": predictions,
            "evaluation": results,
        }


if __name__ == "__main__":

    csv_path = input(
        "Enter CSV file path: "
    ).strip().replace('"', "")

    target_column = input(
        "Enter target column name: "
    ).strip()

    trainer = LightGBMTrainer()

    result = trainer.train(
        csv_path,
        target_column,
    )

    print(
        "\nTraining pipeline completed successfully."
    )
