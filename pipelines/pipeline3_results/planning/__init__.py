"""Typed analytical planning for Pipeline 3."""

from pipelines.pipeline3_results.planning.schemas import (
    AnalyticalPlan,
    GroupingExpression,
    PlanBuildResult,
    PlanFilter,
    PlanProvenance,
    PlanValidation,
    SortExpression,
)


def __getattr__(name: str):
    if name == "build_analytical_plan":
        from pipelines.pipeline3_results.planning.planner import build_analytical_plan

        return build_analytical_plan
    if name == "validate_analytical_plan":
        from pipelines.pipeline3_results.planning.validator import validate_analytical_plan

        return validate_analytical_plan
    if name in {"adapt_plan_to_insight_query", "adapt_plan_to_visualization_request"}:
        from pipelines.pipeline3_results.planning.compatibility import (
            adapt_plan_to_insight_query,
            adapt_plan_to_visualization_request,
        )

        return {
            "adapt_plan_to_insight_query": adapt_plan_to_insight_query,
            "adapt_plan_to_visualization_request": adapt_plan_to_visualization_request,
        }[name]
    raise AttributeError(name)

__all__ = [
    "AnalyticalPlan", "GroupingExpression", "PlanBuildResult", "PlanFilter",
    "PlanProvenance", "PlanValidation", "SortExpression",
    "build_analytical_plan", "validate_analytical_plan",
    "adapt_plan_to_insight_query", "adapt_plan_to_visualization_request",
]
