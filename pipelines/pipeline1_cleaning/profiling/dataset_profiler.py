from typing import Any, Dict

import pandas as pd

from pipelines.pipeline1_cleaning.profiling.column_profiler import (
    profile_column,
)


def profile_dataset(
    dataframe: pd.DataFrame,
    sample_size: int = 5
) -> Dict[str, Any]:
    """
    Create deterministic evidence describing an entire dataset.

    This profiler:
        - records dataset-level structural information
        - detects exact duplicate rows
        - calculates overall missingness
        - profiles every column

    It does NOT:
        - infer dataset domain
        - infer semantic meaning
        - decide whether something is an error
        - recommend cleaning operations
    """

    # --------------------------------------------------------
    # Dataset shape
    # --------------------------------------------------------

    row_count = len(dataframe)
    column_count = len(dataframe.columns)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    total_cells = (
        row_count * column_count
    )

    total_missing = int(
        dataframe.isna().sum().sum()
    )

    overall_missing_percentage = (
        (total_missing / total_cells) * 100
        if total_cells > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Exact duplicate rows
    # --------------------------------------------------------

    duplicate_row_count = int(
        dataframe.duplicated().sum()
    )

    duplicate_row_percentage = (
        (duplicate_row_count / row_count) * 100
        if row_count > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Physical dtype distribution
    # --------------------------------------------------------

    dtype_distribution = {}

    for dtype in dataframe.dtypes:

        dtype_name = str(dtype)

        dtype_distribution[dtype_name] = (
            dtype_distribution.get(
                dtype_name,
                0
            ) + 1
        )

    # --------------------------------------------------------
    # Column profiles
    # --------------------------------------------------------

    column_profiles = []

    for column_name in dataframe.columns:

        profile = profile_column(
            dataframe[column_name],
            sample_size=sample_size
        )

        column_profiles.append(profile)

    # --------------------------------------------------------
    # Dataset profile
    # --------------------------------------------------------

    return {
        "shape": {
            "rows": row_count,
            "columns": column_count
        },

        "column_names": [
            str(column)
            for column in dataframe.columns
        ],

        "dtype_distribution":
            dtype_distribution,

        "missingness": {
            "total_missing_cells":
                total_missing,

            "overall_missing_percentage":
                round(
                    overall_missing_percentage,
                    4
                )
        },

        "duplicates": {
            "exact_duplicate_rows":
                duplicate_row_count,

            "percentage":
                round(
                    duplicate_row_percentage,
                    4
                )
        },

        "column_profiles":
            column_profiles
    }