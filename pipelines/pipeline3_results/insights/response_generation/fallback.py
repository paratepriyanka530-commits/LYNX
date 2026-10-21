"""Deterministic wording built solely from an existing VerifiedResult."""

from __future__ import annotations

from pipelines.pipeline3_results.execution.schemas import VerifiedResult
from pipelines.pipeline3_results.insights.response_generation.context_builder import (
    format_number,
    format_percentage,
)
from pipelines.pipeline3_results.insights.response_generation.schemas import (
    NaturalLanguageResponse,
    ResponseContext,
)
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan


_AGGREGATION_LABELS = {
    "SUM": "total", "MEAN": "average", "MEDIAN": "median",
    "MIN": "minimum", "MAX": "maximum",
}


def _label(context: ResponseContext, column: str | None, default: str) -> str:
    if column is None:
        return default
    return context.semantic_labels.get(column, column.replace("_", " "))


def _coverage_caveats(result: VerifiedResult) -> list[str]:
    caveats = list(result.warnings)
    excluded = result.coverage.excluded_null_rows
    invalid = result.coverage.excluded_invalid_rows
    if result.status == "PARTIAL" or caveats:
        if excluded:
            caveats.append(f"{excluded:,} records were excluded due to missing values.")
        if invalid:
            caveats.append(f"{invalid:,} records were excluded because their values were invalid.")
    return caveats


def _scalar_text(
    context: ResponseContext, plan: AnalyticalPlan, result: VerifiedResult
) -> str:
    expression = plan.measures[0]
    value = format_number(result.scalar)  # type: ignore[arg-type]
    label = _label(context, expression.column, "value")
    if expression.kind == "COUNT_ROWS":
        return f"There are {value} records."
    if expression.kind == "COUNT_DISTINCT":
        return f"There are {value} distinct {label} values."
    if expression.kind == "COUNT_NON_NULL":
        return f"{label.capitalize()} is available for {value} records."
    unit = context.units_or_formats_if_known.get(expression.column or "")
    display_value = f"{value} {unit}" if unit else value
    if expression.kind == "STATISTIC" and expression.operation == "STDDEV":
        return f"The sample standard deviation of {label} is {display_value}."
    if expression.kind == "STATISTIC" and expression.operation == "RANGE":
        return f"The {label} range is {display_value}."
    aggregation = _AGGREGATION_LABELS.get(expression.operation, "computed")
    return f"The {aggregation} {label} is {display_value}."


def _table_text(
    context: ResponseContext, plan: AnalyticalPlan, result: VerifiedResult
) -> str:
    dimension = plan.dimensions[0].column
    expression = plan.measures[0]
    label = _label(context, expression.column, "count")
    rows = result.rows
    if plan.operation == "RANK":
        value = format_number(rows[0]["value"])  # type: ignore[arg-type]
        groups = [str(row[dimension]) for row in rows]
        direction = "highest" if plan.comparison == "HIGHEST" else "lowest"
        if len(groups) > 1:
            return f"{' and '.join(groups)} are tied for the {direction} {label}, with {value} each."
        return f"{groups[0]} has the {direction} {label} at {value}."
    if plan.operation in {"TOP_N", "BOTTOM_N"}:
        direction = "top" if plan.operation == "TOP_N" else "bottom"
        values = ", ".join(
            f"{row[dimension]} ({format_number(row['value'])})"  # type: ignore[arg-type]
            for row in rows
        )
        return f"The {direction} {len(rows)} results by {label} are: {values}."
    if plan.operation == "PART_TO_WHOLE":
        values = ", ".join(
            f"{row[dimension]} contributes {format_percentage(row['share_of_total'])}"  # type: ignore[arg-type]
            for row in rows[:5]
        )
        return f"{values}."
    values = ", ".join(
        f"{row[dimension]} ({format_number(row['value'])})"  # type: ignore[arg-type]
        for row in rows[:5]
    )
    return f"The verified {label} values are: {values}."


def _series_text(
    context: ResponseContext, plan: AnalyticalPlan, result: VerifiedResult
) -> str:
    temporal = plan.temporal[0].column
    label = _label(context, plan.measures[0].column, "value")
    first, last = result.rows[0], result.rows[-1]
    return (
        f"The {label} series starts at {format_number(first['value'])} on "  # type: ignore[arg-type]
        f"{first[temporal]} and ends at {format_number(last['value'])} on {last[temporal]}."  # type: ignore[arg-type]
    )


def _distribution_text(
    context: ResponseContext, plan: AnalyticalPlan, result: VerifiedResult
) -> str:
    column = plan.measures[0].column
    label = _label(context, column, "value")
    dominant = max(result.rows, key=lambda row: int(row["count"]))
    if "bin_start" in dominant:
        return (
            f"The most populated {label} interval is {format_number(dominant['bin_start'])} "  # type: ignore[arg-type]
            f"to {format_number(dominant['bin_end'])}, with {dominant['count']:,} records."  # type: ignore[arg-type]
        )
    return f"{dominant[column]} is the most frequent {label} value, with {dominant['count']:,} records."


def deterministic_fallback(
    context: ResponseContext,
    plan: AnalyticalPlan,
    result: VerifiedResult,
) -> NaturalLanguageResponse:
    caveats = _coverage_caveats(result)
    if result.status == "EMPTY":
        text = "No matching records were found after applying the requested filters."
    elif result.status not in {"READY", "PARTIAL"}:
        text = "The requested result could not be presented safely."
    elif result.result_type in {"SCALAR", "STATISTIC"}:
        text = _scalar_text(context, plan, result)
    elif result.result_type == "TABLE":
        text = _table_text(context, plan, result)
    elif result.result_type == "SERIES":
        text = _series_text(context, plan, result)
    elif result.result_type == "DISTRIBUTION":
        text = _distribution_text(context, plan, result)
    elif result.result_type == "RELATIONSHIP":
        first, second = plan.measures[0].column, plan.measures[1].column
        text = (
            f"The analysis contains {result.coverage.usable_rows:,} valid "
            f"{_label(context, first, 'first value')}–{_label(context, second, 'second value')} pairs."
        )
    else:
        text = "The requested result is not available for explanation."
    if caveats:
        text = f"{text} {' '.join(caveats)}"
    return NaturalLanguageResponse(
        text=text,
        language=context.requested_language,
        status="FALLBACK",
        verified_result_id=result.result_id,
        caveats=caveats,
        fallback_used=True,
    )


__all__ = ["deterministic_fallback"]
