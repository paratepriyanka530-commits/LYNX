"""Presentation adapters from canonical VerifiedResult without recomputation."""

from __future__ import annotations

from pipelines.pipeline3_results.execution.schemas import VerifiedResult
from pipelines.pipeline3_results.insights.schemas import (
    InsightEvidence,
    InsightResult,
    InsightRow,
)
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan
from pipelines.pipeline3_results.visualization.schemas import (
    ChartData,
    ChartPoint,
    HistogramBin,
)


def verified_result_to_insight(
    plan: AnalyticalPlan,
    result: VerifiedResult,
) -> InsightResult | None:
    """Shape shared truth for the existing deterministic Insights response layer."""
    if result.status != "READY":
        return None
    expression = plan.measures[0] if plan.measures else None
    if expression is None or expression.kind == "STATISTIC":
        return None
    aggregation = "COUNT" if expression.kind.startswith("COUNT") else expression.operation
    if aggregation not in {"SUM", "MEAN", "MEDIAN", "MIN", "MAX", "COUNT"}:
        return None
    count_mode = expression.kind if expression.kind.startswith("COUNT") else None
    dimension = plan.dimensions[0].column if plan.dimensions else None
    rows = [
        InsightRow(group=row[dimension], value=row["value"])
        for row in result.rows
        if dimension is not None
    ]
    if plan.operation == "RANK":
        query_type = "GROUPED_COMPARISON"
    elif plan.operation == "TOP_N":
        query_type = "TOP_N"
    elif plan.operation == "BOTTOM_N":
        query_type = "BOTTOM_N"
    elif dimension is not None:
        query_type = "GROUPED_AGGREGATION"
    else:
        query_type = "SIMPLE_AGGREGATION"
    scalar = result.scalar
    winner = rows[0].group if plan.operation == "RANK" and len(rows) == 1 else None
    value = (
        rows[0].value if plan.operation == "RANK" and rows else scalar
    )
    evidence = InsightEvidence(
        total_rows=result.coverage.source_rows,
        usable_rows=result.coverage.usable_rows,
        excluded_null_rows=(result.coverage.excluded_null_rows + result.coverage.excluded_invalid_rows),
        group_count=(len(rows) if dimension is not None else None),
        grouped_values=rows,
        signals=[
            f"verified_result:{result.result_id}",
            f"verification:{result.verification.status}",
            "insight_compatibility:no_recomputation",
        ],
    )
    return InsightResult(
        query_type=query_type,  # type: ignore[arg-type]
        measure_column=expression.column,
        dimension_column=dimension,
        aggregation=aggregation,  # type: ignore[arg-type]
        comparison=plan.comparison,
        count_mode=count_mode,  # type: ignore[arg-type]
        value=value, winner=winner, rows=rows, evidence=evidence, has_data=True,
    )


def verified_result_to_visualization_item(
    plan: AnalyticalPlan,
    result: VerifiedResult,
) -> dict | None:
    """Shape VerifiedResult into existing chart data without recalculating values."""
    if result.status != "READY" or result.result_type in {"SCALAR", "STATISTIC"}:
        return None
    signals = [
        f"verified_result:{result.result_id}",
        f"verification:{result.verification.status}",
        "visualization_compatibility:no_recomputation",
    ]
    chart_type = None
    x_column = ""
    y_column = None
    points: list[ChartPoint] = []
    bins: list[HistogramBin] = []
    aggregation = None
    if result.result_type == "SERIES":
        chart_type = "LINE"
        x_column = plan.temporal[0].column
        y_column = plan.measures[0].column
        aggregation = plan.measures[0].operation.casefold() if plan.measures[0].operation else None
        points = [ChartPoint(x=row[x_column], y=row["value"]) for row in result.rows]
    elif result.result_type == "RELATIONSHIP":
        chart_type = "SCATTER"
        x_column, y_column = plan.measures[0].column, plan.measures[1].column
        points = [ChartPoint(x=row[x_column], y=row[y_column]) for row in result.rows]
    elif result.result_type == "DISTRIBUTION":
        distribution_kind = result.execution_metadata.details.get("distribution_kind")
        if distribution_kind == "numeric_bins":
            chart_type = "HISTOGRAM"
            x_column = plan.measures[0].column
            bins = [HistogramBin(**row) for row in result.rows]
        else:
            chart_type = "COUNT_BAR"
            x_column = plan.measures[0].column
            points = [ChartPoint(x=row[x_column], y=row["count"]) for row in result.rows]
            aggregation = "count"
    elif result.result_type == "TABLE" and plan.dimensions:
        chart_type = "PIE" if plan.operation == "PART_TO_WHOLE" else "BAR"
        x_column = plan.dimensions[0].column
        y_column = plan.measures[0].column
        aggregation = (
            "count" if plan.measures[0].kind.startswith("COUNT")
            else plan.measures[0].operation.casefold() if plan.measures[0].operation else None
        )
        points = [ChartPoint(x=row[x_column], y=row["value"]) for row in result.rows]
    if chart_type is None:
        return None
    chart_data = ChartData(
        chart_type=chart_type,  # type: ignore[arg-type]
        x_column=x_column, y_column=y_column,
        aggregation=aggregation,  # type: ignore[arg-type]
        points=points, bins=bins, signals=signals,
        usable_row_count=result.coverage.usable_rows,
        sampled=False, has_data=bool(points or bins),
    )
    return {
        "rank": 1,
        "chart_type": chart_type,
        "presentation_variant": None,
        "x_column": x_column,
        "y_column": y_column,
        "score": None,
        "chart_data": chart_data.model_dump(mode="json"),
        "signals": signals,
    }


__all__ = ["verified_result_to_insight", "verified_result_to_visualization_item"]
