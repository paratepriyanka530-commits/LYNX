"""Allow-listed deterministic cleaning actions."""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence

import pandas as pd

from pipelines.pipeline1_cleaning.techniques.date_parsing import parse_dates
from pipelines.pipeline1_cleaning.techniques.numeric_coercion import coerce_dtype
from pipelines.pipeline1_cleaning.techniques.missing_values import impute_missing


ActionHandler = Callable[[pd.DataFrame, Sequence[str], Dict[str, Any]], pd.DataFrame]


def no_op(
    dataframe: pd.DataFrame, target_columns: Sequence[str], parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Return the executor-owned dataframe without changing it."""
    if target_columns or parameters:
        raise ValueError("no_op does not accept targets or parameters.")
    return dataframe


def trim_whitespace(
    dataframe: pd.DataFrame, target_columns: Sequence[str], parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Strip only the edges of string values in explicitly targeted columns."""
    if parameters:
        raise ValueError("trim_whitespace does not accept parameters.")
    if not target_columns:
        raise ValueError("trim_whitespace requires at least one target column.")
    for column in target_columns:
        original_dtype = dataframe[column].dtype
        dataframe[column] = dataframe[column].map(
            lambda value: value.strip() if isinstance(value, str) else value,
            na_action="ignore",
        )
        if dataframe[column].dtype != original_dtype:
            dataframe[column] = dataframe[column].astype(original_dtype)
    return dataframe


def normalize_categorical_labels(
    dataframe: pd.DataFrame, target_columns: Sequence[str], parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Apply only an explicit exact-value mapping to one categorical column."""
    if len(target_columns) != 1:
        raise ValueError("normalize_categorical_labels requires exactly one target column.")
    mapping = parameters.get("mapping")
    if set(parameters) != {"mapping"} or not isinstance(mapping, dict) or not mapping:
        raise ValueError("normalize_categorical_labels requires one non-empty mapping.")
    column = target_columns[0]
    original = dataframe[column]
    dataframe[column] = pd.Series(
        [
            mapping[value] if isinstance(value, str) and value in mapping else value
            for value in original.tolist()
        ],
        index=original.index,
        dtype=original.dtype,
    )
    return dataframe


ACTION_REGISTRY: Dict[str, ActionHandler] = {
    "no_op": no_op,
    "trim_whitespace": trim_whitespace,
    "normalize_categorical_labels": normalize_categorical_labels,
    "parse_dates": parse_dates,
    "coerce_dtype": coerce_dtype,
    "impute_missing": impute_missing,
}


def get_action(action_id: str) -> ActionHandler:
    try:
        return ACTION_REGISTRY[action_id]
    except KeyError as exc:
        raise ValueError(f"Unknown action_id: {action_id}") from exc
