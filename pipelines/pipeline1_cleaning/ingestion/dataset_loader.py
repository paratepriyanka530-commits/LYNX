from pathlib import Path
from typing import Dict, Any

import pandas as pd


# ============================================================
# Dataset Loader
# ============================================================

def load_dataset(file_path: str) -> Dict[str, Any]:
    """
    Load a validated dataset into a pandas DataFrame.

    Supported formats:
        .csv
        .xlsx
        .xls

    Important:
        This function assumes that file validation
        has already been completed successfully.

    Returns:
        Dictionary containing:
            - dataframe
            - file_name
            - file_format
            - rows
            - columns
    """

    path = Path(file_path)

    extension = path.suffix.lower()

    # --------------------------------------------------------
    # Load according to format
    # --------------------------------------------------------

    try:

        if extension == ".csv":

            dataframe = pd.read_csv(path)

        elif extension in {".xlsx", ".xls"}:

            dataframe = pd.read_excel(path)

        else:

            raise ValueError(
                f"Unsupported dataset format: {extension}"
            )

    except Exception as exc:

        raise RuntimeError(
            f"Failed to load dataset '{path.name}': {exc}"
        ) from exc

    # --------------------------------------------------------
    # Return loaded dataset information
    # --------------------------------------------------------

    return {
        "dataframe": dataframe,
        "file_name": path.name,
        "file_format": extension,
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
    }


# ============================================================
# Manual Test
# ============================================================

if __name__ == "__main__":

    file_path = (
        r"G:\My Drive\LYNX\tests\datasets\valid\pizza_sales.xlsx"
    )

    result = load_dataset(file_path)

    print("Dataset loaded successfully.")
    print("File:", result["file_name"])
    print("Format:", result["file_format"])
    print("Rows:", result["rows"])
    print("Columns:", result["columns"])

    print("\nPreview:")
    print(result["dataframe"].head())