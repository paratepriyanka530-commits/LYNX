"""
LYNX TabPFN Trainer

Uses the official pretrained TabPFN model.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from . import build_model
from ..artifact_io import training_artifact_dir


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset.
    """

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            "Input file must be a CSV file."
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError(
            "The CSV file is empty."
        )

    return data


def prepare_data(
    data: pd.DataFrame,
    target_column: str,
):
    """
    Separate features and target.
    """

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    X = data.drop(
        columns=[target_column]
    )

    y = data[target_column]

    return X, y


def main(
    csv_path=None,
    target_column=None,
    dataset_id=None,
    run_id=None,
    artifact_root=None,
):

    if csv_path is None:
        csv_path = input(
            "Enter CSV file path: "
        ).strip().replace('"', "")

    if target_column is None:
        target_column = input(
            "Enter target column name: "
        ).strip()

    print("Loading dataset...")

    data = load_csv(csv_path)

    print(
        f"Dataset Shape: {data.shape}"
    )

    X, y = prepare_data(
        data,
        target_column
    )

    print(
        "Performing train/test split..."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(
        f"Training Rows: {len(X_train)}"
    )

    print(
        f"Testing Rows: {len(X_test)}"
    )

    # Reduce training size for TabPFN
    if len(X_train) > 5000:

        X_train = X_train.sample(
            n=5000,
            random_state=42
        )

        y_train = y_train.loc[
            X_train.index
        ]

        print(
            f"TabPFN training set reduced to {len(X_train)} rows."
        )

    # Reduce test size for GPU memory
    if len(X_test) > 1000:

        X_test = X_test.sample(
            n=1000,
            random_state=42
        )

        y_test = y_test.loc[
            X_test.index
        ]

        print(
            f"TabPFN test set reduced to {len(X_test)} rows."
        )

    print(
        "Loading official TabPFN model..."
    )

    model = build_model("classification", random_state=42, device="cuda")

    print(
        "Fitting TabPFN..."
    )

    model.train(
        X_train,
        y_train,
    )

    print(
        "TabPFN fit completed successfully."
    )

    print(
        "Generating predictions..."
    )

    predictions = model.predict(
        X_test
    )

    print(
        f"Predictions generated: {len(predictions)}"
    )

    print(
        "\nSample Predictions:"
    )

    print(
        predictions[:10]
    )

    # ==========================
    # Evaluation Metrics
    # ==========================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
)

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
)
    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(
        "\n===== LYNX TabPFN Evaluation ====="
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    # ==========================
    # Save Model
    # ==========================

    print(
        "\nSaving model..."
    )

    artifact_dir = training_artifact_dir(
        "tabpfn", dataset_id or Path(csv_path).stem, run_id, artifact_root
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(artifact_dir / "tabpfn_model.pkl"))

    print(
        "TabPFN model saved successfully."
    )

    print(
        "\nTraining pipeline completed successfully."
    )

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "artifact_dir": artifact_dir,
    }


if __name__ == "__main__":
    main()
