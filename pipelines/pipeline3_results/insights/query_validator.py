"""Deterministic structural validation for executable insight queries."""

from __future__ import annotations

import pandas as pd

from pipelines.pipeline3_results.insights.schemas import (
    InsightQuery,
    InsightValidation,
)
from pipelines.pipeline3_results.visualization.schemas import DatasetClassification


_NUMERIC_AGGREGATIONS = {"SUM", "MEAN", "MIN", "MAX", "MEDIAN"}
_GROUPABLE_ROLES = {"CATEGORY", "BOOLEAN", "TEMPORAL"}


def validate_insight_query(
    dataframe: pd.DataFrame,
    classification: DatasetClassification,
    query: InsightQuery,
) -> InsightValidation:
    """Validate columns, roles, operations, and required query structure."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if not isinstance(classification, DatasetClassification):
        raise TypeError("classification must be a DatasetClassification")
    if not isinstance(query, InsightQuery):
        raise TypeError("query must be an InsightQuery")

    errors: list[str] = []
    signals = list(query.signals)

    if query.filters:
        errors.append("Filters are not supported by the initial Insights engine.")
    if query.temporal_columns:
        errors.append("Temporal insight operations are not supported by the initial Insights engine.")

    measure = query.measure_column
    dimension = query.dimension_column
    measure_class = classification.get_column(measure) if measure else None
    dimension_class = classification.get_column(dimension) if dimension else None

    if measure is not None and measure not in dataframe.columns:
        errors.append(f"Measure column '{measure}' does not exist.")
    if dimension is not None and dimension not in dataframe.columns:
        errors.append(f"Dimension column '{dimension}' does not exist.")

    if query.aggregation == "NONE":
        errors.append("An executable insight aggregation is required.")
    elif query.aggregation in _NUMERIC_AGGREGATIONS:
        if measure is None:
            errors.append(f"{query.aggregation} requires a measure column.")
        elif measure_class is None or measure_class.role != "MEASURE":
            role = measure_class.role if measure_class else "unclassified"
            errors.append(
                f"{query.aggregation} requires a MEASURE column; '{measure}' is {role}."
            )
    elif query.aggregation == "COUNT":
        if query.count_mode == "COUNT_ROWS":
            if measure is not None:
                errors.append("COUNT_ROWS must not specify a measure column.")
        elif query.count_mode in {"COUNT_NON_NULL", "COUNT_DISTINCT"}:
            if measure is None:
                errors.append(f"{query.count_mode} requires a column.")
        else:
            errors.append("COUNT requires an explicit deterministic count mode.")

    grouped = query.query_type in {"GROUPED_COMPARISON", "TOP_N", "BOTTOM_N"}
    if grouped:
        if dimension is None:
            errors.append(f"{query.query_type} requires a grouping dimension.")
        elif dimension_class is None or dimension_class.role not in _GROUPABLE_ROLES:
            role = dimension_class.role if dimension_class else "unclassified"
            errors.append(
                f"Grouping requires CATEGORY, BOOLEAN, or TEMPORAL; '{dimension}' is {role}."
            )

    if query.query_type == "GROUPED_COMPARISON":
        if query.comparison not in {"HIGHEST", "LOWEST"}:
            errors.append("GROUPED_COMPARISON requires HIGHEST or LOWEST.")
    elif query.query_type == "TOP_N":
        if query.comparison != "TOP_N":
            errors.append("TOP_N query type requires TOP_N comparison.")
        if query.limit is None:
            errors.append("TOP_N requires a positive limit.")
    elif query.query_type == "BOTTOM_N":
        if query.comparison != "BOTTOM_N":
            errors.append("BOTTOM_N query type requires BOTTOM_N comparison.")
        if query.limit is None:
            errors.append("BOTTOM_N requires a positive limit.")
    elif query.query_type == "SIMPLE_AGGREGATION":
        if dimension is not None or query.comparison != "NONE":
            errors.append("SIMPLE_AGGREGATION cannot contain grouping or comparison fields.")

    if len(dataframe) == 0:
        errors.append("The dataframe has zero rows.")

    if errors:
        return InsightValidation(
            status="INVALID",
            failure_code="INVALID_COMBINATION",
            query=query,
            message=errors[0],
            errors=errors,
            signals=[*signals, "insight_validator:invalid"],
        )

    return InsightValidation(
        status="VALID",
        query=query,
        message="Insight query is valid.",
        signals=[*signals, "insight_validator:valid"],
    )


__all__ = ["validate_insight_query"]
