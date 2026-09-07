from pathlib import Path
from typing import Dict, Any

import pandas as pd


# ============================================================
# Allowed File Formats
# ============================================================

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


# ============================================================
# File Validation
# ============================================================

def validate_file(file_path: str) -> Dict[str, Any]:
    """
    Validate an input dataset before it enters Pipeline 1.

    Phase 1 rules:
        1. File must exist.
        2. Path must point to a regular file.
        3. File must not be empty.
        4. Only CSV, XLSX and XLS are accepted.
        5. File must be readable.
        6. Dataset must contain at least 1 row.
        7. Dataset must contain at least 1 column.
        8. No fixed row, column, or file-size limit is imposed here.

    Important:
        This module performs entry validation only.
        It does NOT clean data or make preprocessing decisions.
    """

    path = Path(file_path)

    # --------------------------------------------------------
    # 1. Check existence
    # --------------------------------------------------------

    if not path.exists():
        return {
            "valid": False,
            "reason": "File not found."
        }

    # --------------------------------------------------------
    # 2. Ensure path is a file
    # --------------------------------------------------------

    if not path.is_file():
        return {
            "valid": False,
            "reason": "The supplied path does not point to a file."
        }

    # --------------------------------------------------------
    # 3. Check extension
    # --------------------------------------------------------

    extension = path.suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return {
            "valid": False,
            "reason": (
                f"Unsupported file format '{extension}'. "
                "Only .csv, .xlsx and .xls are allowed."
            )
        }

    # --------------------------------------------------------
    # 4. Check zero-byte file
    # --------------------------------------------------------

    file_size_bytes = path.stat().st_size

    if file_size_bytes == 0:
        return {
            "valid": False,
            "reason": "File is empty."
        }

    # --------------------------------------------------------
    # 5. Attempt safe dataset reading
    # --------------------------------------------------------

    try:

        if extension == ".csv":
            dataframe = pd.read_csv(path)

        else:
            dataframe = pd.read_excel(path)

    except Exception as exc:

        return {
            "valid": False,
            "reason": f"Unable to read dataset: {exc}"
        }

    # --------------------------------------------------------
    # 6. Minimum row requirement
    # --------------------------------------------------------

    row_count = len(dataframe)

    if row_count < 1:
        return {
            "valid": False,
            "reason": (
                "Dataset must contain at least 1 data row."
            )
        }

    # --------------------------------------------------------
    # 7. Minimum column requirement
    # --------------------------------------------------------

    column_count = len(dataframe.columns)

    if column_count < 1:
        return {
            "valid": False,
            "reason": (
                "Dataset must contain at least 1 column."
            )
        }

    # --------------------------------------------------------
    # 8. Detect duplicate column names
    # --------------------------------------------------------

    duplicate_columns = (
        dataframe.columns[
            dataframe.columns.duplicated()
        ]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # 9. Validation passed
    # --------------------------------------------------------

    return {
        "valid": True,

        "file_name": path.name,
        "file_format": extension,

        "file_size_bytes": file_size_bytes,

        "rows": row_count,
        "columns": column_count,

        "duplicate_columns": duplicate_columns,

        "absolute_path": str(path.resolve())
    }


# ============================================================
# Manual Test
# ============================================================

if __name__ == "__main__":

    file_path = r"G:\My Drive\LYNX\tests\datasets\valid\pizza_sales.xlsx"

    result = validate_file(file_path)

    print(result)