"""Bounded, privacy-aware metadata for the Analysis Overview response."""

from __future__ import annotations

from datetime import date, datetime
import math
from typing import Any, Iterable, Mapping

import pandas as pd

from pipelines.pipeline1_cleaning.techniques.privacy_masking import REDACTION_VALUE


SOURCE_STAGE = "original_ingestion"
MAX_PREVIEW_ROWS = 10
MAX_PREVIEW_COLUMNS = 100


def measure_memory_footprint_bytes(dataframe: pd.DataFrame) -> int | None:
    """Return deep Pandas memory usage for the ingestion frame, index included."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    try:
        return int(dataframe.memory_usage(index=True, deep=True).sum())
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def serialize_json_cell(value: Any) -> Any:
    """Convert one ingested scalar to strict JSON without changing the frame."""
    if value is None:
        return None
    try:
        missing = pd.isna(value)
        if not hasattr(missing, "__len__") and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): serialize_json_cell(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_json_cell(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return serialize_json_cell(item())
        except (TypeError, ValueError, OverflowError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def build_raw_data_preview(
    dataframe: pd.DataFrame,
    *,
    privacy_clearance_passed: bool,
    pii_columns: Iterable[str] = (),
    row_limit: int = MAX_PREVIEW_ROWS,
    column_limit: int = MAX_PREVIEW_COLUMNS,
) -> dict[str, Any]:
    """Build an ordered preview without mutating or returning the source frame."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if row_limit < 0 or row_limit > MAX_PREVIEW_ROWS:
        raise ValueError(f"row_limit must be between 0 and {MAX_PREVIEW_ROWS}")
    if column_limit < 1 or column_limit > MAX_PREVIEW_COLUMNS:
        raise ValueError(f"column_limit must be between 1 and {MAX_PREVIEW_COLUMNS}")

    if not privacy_clearance_passed:
        return {
            "available": False,
            "unavailable_reason": "privacy_clearance_not_passed",
            "source_stage": SOURCE_STAGE,
            "row_limit": row_limit,
            "column_limit": column_limit,
            "total_row_count": int(len(dataframe)),
            "total_column_count": int(len(dataframe.columns)),
            "columns_truncated": False,
            "columns": [],
            "rows": [],
            "returned_row_count": 0,
            "privacy_redacted_columns": [],
        }

    selected = dataframe.iloc[:row_limit, :column_limit]
    columns = [str(column) for column in selected.columns]
    protected = {str(column) for column in pii_columns}
    redacted_positions = {
        index for index, column in enumerate(columns) if column in protected
    }
    rows: list[list[Any]] = []
    for raw_row in selected.itertuples(index=False, name=None):
        serialized: list[Any] = []
        for index, value in enumerate(raw_row):
            safe_value = serialize_json_cell(value)
            serialized.append(
                REDACTION_VALUE
                if index in redacted_positions and safe_value is not None
                else safe_value
            )
        rows.append(serialized)

    return {
        "available": True,
        "unavailable_reason": None,
        "source_stage": SOURCE_STAGE,
        "row_limit": row_limit,
        "column_limit": column_limit,
        "total_row_count": int(len(dataframe)),
        "total_column_count": int(len(dataframe.columns)),
        "columns_truncated": len(dataframe.columns) > len(columns),
        "columns": columns,
        "rows": rows,
        "returned_row_count": len(rows),
        "privacy_redacted_columns": [
            column for index, column in enumerate(columns) if index in redacted_positions
        ],
    }


def build_dataset_overview(
    dataframe: pd.DataFrame,
    *,
    privacy_clearance_passed: bool,
    pii_columns: Iterable[str] = (),
) -> dict[str, Any]:
    """Build additive Overview metadata for the original ingestion stage."""
    return {
        "memory_footprint_bytes": measure_memory_footprint_bytes(dataframe),
        "memory_footprint_stage": SOURCE_STAGE,
        "raw_data_preview": build_raw_data_preview(
            dataframe,
            privacy_clearance_passed=privacy_clearance_passed,
            pii_columns=pii_columns,
        ),
    }


def unavailable_dataset_overview(reason: str) -> dict[str, Any]:
    """Return a schema-stable unavailable state without failing the core result."""
    return {
        "memory_footprint_bytes": None,
        "memory_footprint_stage": SOURCE_STAGE,
        "raw_data_preview": {
            "available": False,
            "unavailable_reason": reason,
            "source_stage": SOURCE_STAGE,
            "row_limit": MAX_PREVIEW_ROWS,
            "column_limit": MAX_PREVIEW_COLUMNS,
            "total_row_count": 0,
            "total_column_count": 0,
            "columns_truncated": False,
            "columns": [],
            "rows": [],
            "returned_row_count": 0,
            "privacy_redacted_columns": [],
        },
    }


__all__ = [
    "MAX_PREVIEW_COLUMNS",
    "MAX_PREVIEW_ROWS",
    "SOURCE_STAGE",
    "build_dataset_overview",
    "build_raw_data_preview",
    "measure_memory_footprint_bytes",
    "serialize_json_cell",
    "unavailable_dataset_overview",
]
