"""
LYNX CatBoost Trainer

Responsibilities:
    - Load dataset
    - Prepare features and target
    - Preprocess data
    - Train/Test split
    - Train CatBoost model
    - Evaluate performance
    - Save model and preprocessor
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .catboost_model import LYNXCatBoost
from .catboost_preprocessor import CatBoostPreprocessor
from .catboost_evaluator import CatBoostEvaluator

from .data_loader import (
    load_csv,
    prepare_data,
)
from ..artifact_io import training_artifact_dir


class CatBoostTrainer:
    """
    Handles CatBoost training workflow.
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

        print("Loading dataset...")

        data = load_csv(csv_path)

        print(
            f"Dataset Shape: {data.shape}"
        )

        X, y = prepare_data(
            data,
            target_column,
        )

        print(
            "Preprocessing dataset..."
        )

        if not pd.api.types.is_numeric_dtype(y):

            target_encoder = LabelEncoder()

            y = target_encoder.fit_transform(y)

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

        preprocessor = CatBoostPreprocessor()
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        print(
            f"Training Rows: {len(X_train)}"
        )

        print(
            f"Testing Rows: {len(X_test)}"
        )

        model = LYNXCatBoost()

        print(
            "Training CatBoost..."
        )

        model.train(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        results = CatBoostEvaluator.evaluate(
            y_test,
            predictions,
        )

        CatBoostEvaluator.print_results(
            results
        )

        print(
            "Saving model..."
        )

        artifact_dir = training_artifact_dir(
            "catboost", dataset_id or Path(csv_path).stem, run_id, artifact_root
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model.model, artifact_dir / "catboost_model.pkl")
        joblib.dump(preprocessor, artifact_dir / "catboost_preprocessor.pkl")

        print(
            "CatBoost model saved successfully."
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
    )

    target_column = input(
        "Enter target column name: "
    )

    trainer = CatBoostTrainer()

    trainer.train(
        csv_path,
        target_column,
    )

    print(
        "\nTraining pipeline completed successfully."
    )
    
