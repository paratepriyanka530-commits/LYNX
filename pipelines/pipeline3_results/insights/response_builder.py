"""Build factual frontend-safe Insight responses from computed values only."""

from __future__ import annotations

from pipelines.pipeline3_results.insights.schemas import (
    InsightResponse,
    InsightResult,
)


_AGGREGATION_LABELS = {
    "SUM": "total",
    "MEAN": "average",
    "COUNT": "count",
    "MIN": "minimum",
    "MAX": "maximum",
    "MEDIAN": "median",
}


def _format_value(value: float | int) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.6g}"


def _answer(result: InsightResult) -> str:
    label = _AGGREGATION_LABELS[result.aggregation]
    if result.query_type == "SIMPLE_AGGREGATION":
        assert result.value is not None
        if result.count_mode == "COUNT_ROWS":
            return f"The dataset contains {_format_value(result.value)} rows."
        if result.count_mode == "COUNT_DISTINCT":
            return (
                f"There are {_format_value(result.value)} distinct "
                f"{result.measure_column} values."
            )
        if result.count_mode == "COUNT_NON_NULL":
            return (
                f"There are {_format_value(result.value)} non-null "
                f"{result.measure_column} values."
            )
        return (
            f"The {label} {result.measure_column} is "
            f"{_format_value(result.value)}."
        )

    if result.query_type == "GROUPED_COMPARISON":
        assert result.winner is not None and result.value is not None
        direction = "highest" if result.comparison == "HIGHEST" else "lowest"
        measure_phrase = (
            label
            if result.aggregation == "COUNT" and result.measure_column is None
            else f"{label} {result.measure_column}"
        )
        return (
            f"{result.winner} has the {direction} {measure_phrase} "
            f"at {_format_value(result.value)}."
        )

    direction = "top" if result.query_type == "TOP_N" else "bottom"
    values = ", ".join(
        f"{row.group} ({_format_value(row.value)})" for row in result.rows
    )
    return (
        f"The {direction} {len(result.rows)} {result.dimension_column} groups by "
        f"{label} {result.measure_column} are: {values}."
    )


def build_insight_response(
    result: InsightResult,
    *,
    signals: list[str] | None = None,
) -> InsightResponse:
    """Create a response whose language is derived exclusively from result data."""
    if not isinstance(result, InsightResult):
        raise TypeError("result must be an InsightResult")
    combined_signals = [*(signals or []), "insight_response:factual_answer_built"]
    if not result.has_data:
        return InsightResponse(
            status="INVALID",
            answer=None,
            result=result,
            evidence=result.evidence,
            message="No usable data remained for the requested computation.",
            errors=["No usable non-null values were available."],
            signals=[*combined_signals, "insight_response:no_usable_data"],
        )
    return InsightResponse(
        status="READY",
        answer=_answer(result),
        result=result,
        evidence=result.evidence,
        message="Insight computed deterministically from dataset values.",
        signals=combined_signals,
    )


def build_controlled_response(
    status: str,
    message: str,
    *,
    clarification_options: list[str] | None = None,
    errors: list[str] | None = None,
    signals: list[str] | None = None,
    failure_code: str | None = None,
) -> InsightResponse:
    """Build clarification/invalid responses without fabricated result data."""
    return InsightResponse(
        status=status,  # type: ignore[arg-type]
        failure_code=failure_code,  # type: ignore[arg-type]
        message=message,
        clarification_options=clarification_options or [],
        errors=errors or [],
        signals=signals or [],
    )


__all__ = ["build_controlled_response", "build_insight_response"]
