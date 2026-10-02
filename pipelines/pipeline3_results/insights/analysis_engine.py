"""Pandas-only deterministic computation engine for validated Insights queries."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from pipelines.pipeline3_results.insights.schemas import (
    InsightEvidence,
    InsightQuery,
    InsightResult,
    InsightRow,
)


def _python_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _numeric_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.notna() & numeric.map(
        lambda value: math.isfinite(float(value)) if pd.notna(value) else False
    )
    return numeric, finite


def _aggregate(series: pd.Series, aggregation: str) -> float | int:
    if aggregation == "SUM":
        return float(series.sum())
    if aggregation == "MEAN":
        return float(series.mean())
    if aggregation == "MEDIAN":
        return float(series.median())
    if aggregation == "MIN":
        return float(series.min())
    if aggregation == "MAX":
        return float(series.max())
    raise ValueError(f"Unsupported numeric aggregation: {aggregation}")


def _simple_result(dataframe: pd.DataFrame, query: InsightQuery) -> InsightResult:
    total = len(dataframe)
    signals: list[str] = []

    if query.aggregation == "COUNT" and query.count_mode == "COUNT_ROWS":
        value: float | int = total
        usable = total
        signals.append("counted all dataframe rows")
    elif query.aggregation == "COUNT":
        assert query.measure_column is not None
        series = dataframe[query.measure_column]
        valid = series.notna()
        usable = int(valid.sum())
        if query.count_mode == "COUNT_DISTINCT":
            value = int(series[valid].nunique(dropna=True))
            signals.append(f"counted distinct non-null values of {query.measure_column}")
        else:
            value = usable
            signals.append(f"counted non-null values of {query.measure_column}")
    else:
        assert query.measure_column is not None
        numeric, finite = _numeric_series(dataframe[query.measure_column])
        usable = int(finite.sum())
        if usable:
            value = _aggregate(numeric[finite], query.aggregation)
            signals.append(
                f"computed {query.aggregation} from {usable} usable {query.measure_column} values"
            )
        else:
            value = None  # type: ignore[assignment]
            signals.append("no usable numeric values")

    excluded = total - usable
    evidence = InsightEvidence(
        total_rows=total,
        usable_rows=usable,
        excluded_null_rows=excluded,
        signals=[
            *signals,
            f"{excluded} rows excluded from the computation",
        ],
    )
    return InsightResult(
        query_type=query.query_type,
        measure_column=query.measure_column,
        dimension_column=None,
        aggregation=query.aggregation,
        comparison=query.comparison,
        count_mode=query.count_mode,
        value=value,
        evidence=evidence,
        has_data=value is not None,
    )


def _grouped_values(
    dataframe: pd.DataFrame,
    query: InsightQuery,
) -> tuple[list[InsightRow], int, int, list[str]]:
    assert query.dimension_column is not None
    dimension = query.dimension_column
    total = len(dataframe)
    signals: list[str] = []

    if query.aggregation == "COUNT" and query.count_mode == "COUNT_ROWS":
        valid = dataframe[dimension].notna()
        working = dataframe.loc[valid, [dimension]]
        grouped = working.groupby(dimension, dropna=True, sort=False).size()
        usable = len(working)
        signals.append(f"counted rows within each {dimension} group")
    elif query.aggregation == "COUNT":
        assert query.measure_column is not None
        measure = query.measure_column
        valid = dataframe[dimension].notna() & dataframe[measure].notna()
        working = dataframe.loc[valid, [dimension, measure]]
        grouped_obj = working.groupby(dimension, dropna=True, sort=False)[measure]
        if query.count_mode == "COUNT_DISTINCT":
            grouped = grouped_obj.nunique(dropna=True)
            signals.append(f"counted distinct {measure} values within each {dimension} group")
        else:
            grouped = grouped_obj.count()
            signals.append(f"counted non-null {measure} values within each {dimension} group")
        usable = len(working)
    else:
        assert query.measure_column is not None
        measure = query.measure_column
        numeric, finite = _numeric_series(dataframe[measure])
        valid = dataframe[dimension].notna() & finite
        working = pd.DataFrame({dimension: dataframe.loc[valid, dimension], measure: numeric[valid]})
        grouped_obj = working.groupby(dimension, dropna=True, sort=False)[measure]
        operations = {
            "SUM": grouped_obj.sum,
            "MEAN": grouped_obj.mean,
            "MEDIAN": grouped_obj.median,
            "MIN": grouped_obj.min,
            "MAX": grouped_obj.max,
        }
        grouped = operations[query.aggregation]()
        usable = len(working)
        signals.append(f"computed {query.aggregation} {measure} within each {dimension} group")

    rows = [
        InsightRow(
            group=_python_scalar(group),
            value=(int(value) if query.aggregation == "COUNT" else float(value)),
        )
        for group, value in grouped.items()
        if pd.notna(value) and np.isfinite(float(value))
    ]
    descending = query.comparison in {"HIGHEST", "TOP_N"}
    rows.sort(
        key=lambda row: (
            -float(row.value) if descending else float(row.value),
            str(row.group).casefold(),
        )
    )
    return rows, usable, total - usable, signals


def _grouped_result(dataframe: pd.DataFrame, query: InsightQuery) -> InsightResult:
    all_rows, usable, excluded, signals = _grouped_values(dataframe, query)
    if query.comparison in {"HIGHEST", "LOWEST"}:
        selected = all_rows[:1]
    else:
        selected = all_rows[: query.limit]

    evidence = InsightEvidence(
        total_rows=len(dataframe),
        usable_rows=usable,
        excluded_null_rows=excluded,
        group_count=len(all_rows),
        grouped_values=all_rows,
        signals=[
            *signals,
            f"{excluded} rows excluded from grouped computation",
            f"{len(all_rows)} groups computed",
        ],
    )
    return InsightResult(
        query_type=query.query_type,
        measure_column=query.measure_column,
        dimension_column=query.dimension_column,
        aggregation=query.aggregation,
        comparison=query.comparison,
        count_mode=query.count_mode,
        value=selected[0].value if len(selected) == 1 else None,
        winner=selected[0].group if len(selected) == 1 else None,
        rows=selected,
        evidence=evidence,
        has_data=bool(selected),
    )


def execute_insight_query(
    dataframe: pd.DataFrame,
    query: InsightQuery,
) -> InsightResult:
    """Execute one validated query with fixed pandas operations only."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if not isinstance(query, InsightQuery):
        raise TypeError("query must be an InsightQuery")
    if query.query_type == "SIMPLE_AGGREGATION":
        return _simple_result(dataframe, query)
    return _grouped_result(dataframe, query)


__all__ = ["execute_insight_query"]
