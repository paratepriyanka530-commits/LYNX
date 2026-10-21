"""Temporary adapters from AnalyticalPlan to current downstream contracts."""

from __future__ import annotations

from pipelines.pipeline3_results.insights.schemas import (
    InsightAdaptation,
    InsightQuery,
)
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan
from pipelines.pipeline3_results.visualization.nlp_adapter import VisualizationAdaptation
from pipelines.pipeline3_results.visualization.schemas import VisualizationRequest


_LEGACY_AGGREGATIONS = {"SUM", "MEAN", "MEDIAN", "MIN", "MAX"}


def _unsupported_insight(plan: AnalyticalPlan, message: str, signal: str) -> InsightAdaptation:
    return InsightAdaptation(
        status="NEEDS_CLARIFICATION",
        failure_code="UNSUPPORTED_OPERATION",
        message=message,
        signals=[f"analytical_plan:{plan.plan_id}", signal, "PLAN_READY", "LEGACY_EXECUTION_UNSUPPORTED"],
    )


def adapt_plan_to_insight_query(
    plan: AnalyticalPlan,
    *,
    original_text: str = "",
) -> InsightAdaptation:
    """Map only expressions the current deterministic Insights engine supports."""
    if not isinstance(plan, AnalyticalPlan):
        raise TypeError("plan must be an AnalyticalPlan")
    if plan.filters:
        return _unsupported_insight(plan, "The plan is valid, but legacy Insights cannot execute typed filters.", "plan_adapter:unsupported_filters")
    if plan.temporal:
        return _unsupported_insight(plan, "The plan is valid, but legacy Insights cannot execute temporal grouping.", "plan_adapter:unsupported_temporal")
    if plan.operation in {"TREND", "DISTRIBUTION", "RELATIONSHIP", "STATISTIC", "EXPLORATORY"}:
        return _unsupported_insight(plan, f"The plan is valid, but legacy Insights cannot execute {plan.operation}.", f"plan_adapter:unsupported_operation:{plan.operation}")
    if len(plan.measures) != 1 or len(plan.dimensions) > 1:
        return _unsupported_insight(plan, "Legacy Insights requires exactly one measure and at most one dimension.", "plan_adapter:unsupported_cardinality")

    expression = plan.measures[0]
    measure_column = expression.column
    aggregation = "NONE"
    count_mode = None
    if expression.kind == "AGGREGATION" and expression.operation in _LEGACY_AGGREGATIONS:
        aggregation = expression.operation
    elif expression.kind == "COUNT_ROWS":
        aggregation = "COUNT"
        count_mode = "COUNT_ROWS"
        measure_column = None
    elif expression.kind == "COUNT_NON_NULL":
        aggregation = "COUNT"
        count_mode = "COUNT_NON_NULL"
    elif expression.kind == "COUNT_DISTINCT":
        aggregation = "COUNT"
        count_mode = "COUNT_DISTINCT"
    else:
        return _unsupported_insight(plan, f"Legacy Insights cannot execute {expression.kind}.", f"plan_adapter:unsupported_expression:{expression.kind}")

    query_type = {
        "TOP_N": "TOP_N",
        "BOTTOM_N": "BOTTOM_N",
        "RANK": "GROUPED_COMPARISON",
    }.get(plan.operation, "SIMPLE_AGGREGATION")
    dimension = plan.dimensions[0].column if plan.dimensions else None
    query = InsightQuery(
        original_text=original_text,
        query_type=query_type,  # type: ignore[arg-type]
        measure_column=measure_column,
        dimension_column=dimension,
        aggregation=aggregation,  # type: ignore[arg-type]
        comparison=plan.comparison,
        limit=plan.limit,
        count_mode=count_mode,  # type: ignore[arg-type]
        signals=[f"analytical_plan:{plan.plan_id}", "plan_adapter:insight_query_mapped"],
    )
    return InsightAdaptation(
        status="READY",
        query=query,
        message="Analytical plan is compatible with legacy Insights execution.",
        signals=list(query.signals),
    )


def _unsupported_visualization(
    plan: AnalyticalPlan,
    message: str,
    signal: str,
) -> VisualizationAdaptation:
    return VisualizationAdaptation(
        status="NEEDS_CLARIFICATION",
        failure_code="UNSUPPORTED_OPERATION",
        message=message,
        signals=[f"analytical_plan:{plan.plan_id}", signal, "PLAN_READY", "LEGACY_EXECUTION_UNSUPPORTED"],
    )


def adapt_plan_to_visualization_request(
    plan: AnalyticalPlan,
    *,
    raw_request: str = "",
) -> VisualizationAdaptation:
    """Map a plan without selecting a chart family or changing its semantics."""
    if not isinstance(plan, AnalyticalPlan):
        raise TypeError("plan must be an AnalyticalPlan")
    if plan.filters:
        return _unsupported_visualization(plan, "The plan is valid, but requested visualization execution cannot apply typed filters.", "plan_adapter:unsupported_filters")
    if any(item.kind == "TIME_BUCKET" for item in plan.temporal):
        return _unsupported_visualization(plan, "The plan is valid, but requested visualization execution cannot apply time buckets.", "plan_adapter:unsupported_time_bucket")
    if any(item.kind in {"STATISTIC", "COUNT_DISTINCT", "COUNT_NON_NULL"} for item in plan.measures):
        kind = next(item.kind for item in plan.measures if item.kind in {"STATISTIC", "COUNT_DISTINCT", "COUNT_NON_NULL"})
        return _unsupported_visualization(plan, f"The plan is valid, but requested visualization execution cannot execute {kind}.", f"plan_adapter:unsupported_expression:{kind}")
    if plan.operation in {"RANK", "TOP_N", "BOTTOM_N", "STATISTIC", "EXPLORATORY", "AGGREGATE"}:
        return _unsupported_visualization(plan, f"The plan is valid, but requested visualization execution cannot execute {plan.operation} without changing meaning.", f"plan_adapter:unsupported_operation:{plan.operation}")

    x_column = None
    y_column = None
    intent = None
    aggregation = None
    if plan.operation == "RELATIONSHIP" and len(plan.measures) == 2:
        x_column, y_column = plan.measures[0].column, plan.measures[1].column
        intent = "RELATIONSHIP"
    elif plan.operation == "DISTRIBUTION" and len(plan.measures) == 1:
        x_column = plan.measures[0].column
        intent = "DISTRIBUTION"
    elif plan.operation == "TREND" and plan.temporal and plan.measures:
        x_column, y_column = plan.temporal[0].column, plan.measures[0].column
        intent = "TREND"
        aggregation = plan.measures[0].operation
    elif plan.operation in {"GROUP_AGGREGATE", "PART_TO_WHOLE", "COUNT"} and plan.dimensions:
        x_column = plan.dimensions[0].column
        y_column = plan.measures[0].column if plan.measures and plan.measures[0].kind != "COUNT_ROWS" else None
        intent = "COMPARISON"
        aggregation = "COUNT" if plan.measures and plan.measures[0].kind == "COUNT_ROWS" else plan.measures[0].operation
    else:
        return _unsupported_visualization(plan, "The plan does not map to the current requested-visualization contract.", "plan_adapter:unsupported_shape")

    requested_columns = [item for item in (x_column, y_column) if item is not None]
    request = VisualizationRequest(
        raw_request=raw_request,
        requested_chart_type=None,
        is_explicit_chart_type=False,
        x_column=x_column,
        y_column=y_column,
        aggregation=aggregation,  # type: ignore[arg-type]
        intent=intent,  # type: ignore[arg-type]
        requested_columns=requested_columns,
        parser_signals=[
            f"analytical_plan:{plan.plan_id}",
            "plan_adapter:visualization_request_mapped",
            "plan_adapter:no_chart_selected",
        ],
    )
    return VisualizationAdaptation(
        status="READY",
        request=request,
        message="Analytical plan is compatible with requested visualization execution.",
        signals=list(request.parser_signals),
    )


__all__ = ["adapt_plan_to_insight_query", "adapt_plan_to_visualization_request"]
