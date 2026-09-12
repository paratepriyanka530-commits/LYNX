"""
LYNX XGBoost Trainer

Responsibilities:
    - Load dataset
    - Prepare features and target
    - Preprocess data
    - Train/Test split
    - Train XGBoost model
    - Generate predictions
    - Evaluate model performance
    - Save model
"""

from pathlib import Path

import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .xgboost_model import (
    LYNXXGBoost,
    load_csv,
    prepare_data,
)

from .xgboost_preprocessor import XGBoostPreprocessor
from ..artifact_io import training_artifact_dir
from .xgboost_evaluator import XGBoostEvaluator


class XGBoostTrainer:

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ):

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

        # ------------------------------
        # Prepare Features and Target
        # ------------------------------

        X, y = prepare_data(
            data,
            target_column,
        )

        if y.dtype == "object":

            target_encoder = LabelEncoder()

            y = target_encoder.fit_transform(
                y.astype(str)
            )

        # ------------------------------
        # Train/Test Split
        # ------------------------------

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

        print("Preprocessing training data and transforming held-out data...")
        preprocessor = XGBoostPreprocessor()
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        print(
            f"Training Rows: {len(X_train)}"
        )

        print(
            f"Testing Rows: {len(X_test)}"
        )

        # ------------------------------
        # Model Training
        # ------------------------------

        model = LYNXXGBoost()

        print(
            "Training XGBoost..."
        )

        model.train(
            X_train,
            y_train,
        )

        # ------------------------------
        # Save Model
        # ------------------------------

        artifact_dir = training_artifact_dir(
            "xgboost", dataset_id or Path(csv_path).stem, run_id, artifact_root
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model.model, artifact_dir / "xgboost_model.pkl")
        joblib.dump(preprocessor, artifact_dir / "xgboost_preprocessor.pkl")

        print(
            "Model saved successfully."
        )

        # ------------------------------
        # Prediction
        # ------------------------------

        predictions = model.predict(
            X_test
        )

        # ------------------------------
        # Evaluation
        # ------------------------------

        results = XGBoostEvaluator.evaluate(
            y_test,
            predictions
        )

        XGBoostEvaluator.print_results(
            results
        )

        print(
            "XGBoost training completed successfully."
        )

        return {
            "model": model,
            "preprocessor": preprocessor,
            "artifact_dir": artifact_dir,
            "X_train_raw": X_train_raw,
            "X_test_raw": X_test_raw,
            "X_test": X_test,
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

    trainer = XGBoostTrainer()

    trainer.train(
        csv_path,
        target_column,
    )

    print(
        "\nTraining pipeline completed successfully."
    )
