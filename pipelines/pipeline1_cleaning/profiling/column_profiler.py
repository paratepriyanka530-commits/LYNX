from datetime import date, datetime, time
from typing import Any, Dict

import numpy as np
import pandas as pd


# ============================================================
# Helper Functions
# ============================================================

def make_serializable(value: Any) -> Any:
    """
    Convert Pandas, NumPy, and Python date/time values
    into JSON-safe Python values.
    """

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    # --------------------------------------------------------
    # NumPy values
    # --------------------------------------------------------

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    # --------------------------------------------------------
    # Pandas values
    # --------------------------------------------------------

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Timedelta):
        return str(value)

    # --------------------------------------------------------
    # Native Python date/time values
    # --------------------------------------------------------

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    # --------------------------------------------------------
    # Normal JSON-safe values
    # --------------------------------------------------------

    if isinstance(
        value,
        (str, int, float, bool)
    ):
        return value

    # --------------------------------------------------------
    # Safe fallback
    # --------------------------------------------------------

    return str(value)


def get_sample_values(
    series: pd.Series,
    sample_size: int = 5
) -> list:
    """
    Return a small set of non-null observed values.

    Samples are evidence only.
    They are not used here to infer semantic meaning.
    """

    non_null = series.dropna()

    if non_null.empty:
        return []

    samples = (
        non_null
        .drop_duplicates()
        .head(sample_size)
        .tolist()
    )

    return [
        make_serializable(value)
        for value in samples
    ]


# ============================================================
# Column Profiler
# ============================================================

def profile_column(
    series: pd.Series,
    sample_size: int = 5
) -> Dict[str, Any]:
    """
    Produce deterministic evidence for one dataset column.

    This function describes what is physically observed.
    It does NOT infer semantic meaning and does NOT
    recommend cleaning operations.
    """

    row_count = len(series)

    # --------------------------------------------------------
    # Missingness
    # --------------------------------------------------------

    missing_count = int(
        series.isna().sum()
    )

    non_null_count = (
        row_count - missing_count
    )

    missing_percentage = (
        (missing_count / row_count) * 100
        if row_count > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Uniqueness
    # --------------------------------------------------------

    unique_count = int(
        series.nunique(dropna=True)
    )

    unique_ratio = (
        unique_count / non_null_count
        if non_null_count > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Repetition
    # --------------------------------------------------------

    value_counts = series.value_counts(
        dropna=True
    )

    repeated_values = value_counts[
        value_counts > 1
    ]

    repeated_unique_count = int(
        len(repeated_values)
    )

    repeated_occurrences = int(
        repeated_values.sum()
    )

    repetition_rate = (
        repeated_occurrences / non_null_count
        if non_null_count > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Base Profile
    # --------------------------------------------------------

    profile = {
        "column_name": str(series.name),

        "physical_dtype": str(
            series.dtype
        ),

        "row_count": row_count,

        "non_null_count": non_null_count,

        "missing": {
            "count": missing_count,
            "percentage": round(
                missing_percentage,
                4
            )
        },

        "uniqueness": {
            "unique_count": unique_count,
            "unique_ratio": round(
                unique_ratio,
                6
            )
        },

        "repetition": {
            "repeated_unique_count":
                repeated_unique_count,

            "repeated_occurrences":
                repeated_occurrences,

            "repetition_rate": round(
                repetition_rate,
                6
            )
        },

        "sample_values": get_sample_values(
            series,
            sample_size
        )
    }

    # --------------------------------------------------------
    # Numeric Statistics
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(
        series
    ):

        numeric_series = pd.to_numeric(
            series,
            errors="coerce"
        )

        finite_series = numeric_series[
            np.isfinite(numeric_series)
        ]

        infinity_count = int(
            np.isinf(numeric_series).sum()
        )

        if not finite_series.empty:

            profile["numeric_statistics"] = {
                "min": make_serializable(
                    finite_series.min()
                ),

                "max": make_serializable(
                    finite_series.max()
                ),

                "mean": make_serializable(
                    finite_series.mean()
                ),

                "median": make_serializable(
                    finite_series.median()
                ),

                "std": make_serializable(
                    finite_series.std()
                ),

                "infinity_count":
                    infinity_count
            }

        else:

            profile["numeric_statistics"] = {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "infinity_count":
                    infinity_count
            }

    return profile