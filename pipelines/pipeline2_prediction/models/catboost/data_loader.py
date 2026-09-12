from pathlib import Path
import pandas as pd


def load_csv(csv_path: str) -> pd.DataFrame:

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

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    X = data.drop(
        columns=[target_column]
    )

    y = data[target_column]

    return X, y
