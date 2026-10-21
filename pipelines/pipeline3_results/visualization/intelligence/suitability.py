"""Deterministic chart-family suitability rules over verified truth."""

from __future__ import annotations

from pipelines.pipeline3_results.execution.schemas import VerifiedResult
from pipelines.pipeline3_results.grounding.schemas import DatasetKnowledge
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan
from .schemas import ChartFamily, SuitabilityAssessment, VisualizationPurpose


def purpose_for(plan: AnalyticalPlan, result: VerifiedResult) -> VisualizationPurpose:
    if plan.operation in {"RANK", "TOP_N", "BOTTOM_N"}:
        return VisualizationPurpose.RANKING
    if plan.operation == "TREND" or result.result_type == "SERIES":
        return VisualizationPurpose.TREND
    if plan.operation == "DISTRIBUTION" or result.result_type == "DISTRIBUTION":
        return VisualizationPurpose.DISTRIBUTION
    if plan.operation == "RELATIONSHIP" or result.result_type == "RELATIONSHIP":
        return VisualizationPurpose.RELATIONSHIP
    if plan.operation == "PART_TO_WHOLE":
        return VisualizationPurpose.PART_TO_WHOLE
    return VisualizationPurpose.COMPARISON


def default_family(purpose: VisualizationPurpose) -> ChartFamily:
    return {
        VisualizationPurpose.TREND: ChartFamily.LINE,
        VisualizationPurpose.DISTRIBUTION: ChartFamily.HISTOGRAM,
        VisualizationPurpose.RELATIONSHIP: ChartFamily.SCATTER,
        VisualizationPurpose.PART_TO_WHOLE: ChartFamily.PIE,
    }.get(purpose, ChartFamily.BAR)


def assess_suitability(
    plan: AnalyticalPlan,
    result: VerifiedResult,
    knowledge: DatasetKnowledge,
    family: ChartFamily,
) -> SuitabilityAssessment:
    del knowledge  # eligibility was already enforced by grounding/planning
    purpose = purpose_for(plan, result)
    categories = len(result.rows)
    invalid: str | None = None
    if family == ChartFamily.SCATTER and result.result_type != "RELATIONSHIP":
        invalid = "SCATTER_REQUIRES_NUMERIC_RELATIONSHIP"
    elif family in {ChartFamily.LINE, ChartFamily.AREA} and result.result_type != "SERIES":
        invalid = "LINE_REQUIRES_ORDERED_TEMPORAL_RESULT"
    elif family == ChartFamily.HISTOGRAM and not (
        result.result_type == "DISTRIBUTION"
        and result.execution_metadata.details.get("distribution_kind") == "numeric_bins"
    ):
        invalid = "HISTOGRAM_REQUIRES_NUMERIC_DISTRIBUTION"
    elif family in {ChartFamily.PIE, ChartFamily.DOUGHNUT}:
        expression = plan.measures[0] if plan.measures else None
        additive_table = bool(
            result.result_type == "TABLE" and plan.dimensions and expression
            and (expression.kind.startswith("COUNT") or expression.operation == "SUM")
        )
        if purpose != VisualizationPurpose.PART_TO_WHOLE and not additive_table:
            invalid = "PART_TO_WHOLE_RESULT_REQUIRED"
    if invalid:
        return SuitabilityAssessment(status="INVALID", reason_codes=[invalid], message="The requested chart family is not compatible with the verified analytical result.")
    if family in {ChartFamily.PIE, ChartFamily.DOUGHNUT} and categories > 30:
        return SuitabilityAssessment(status="NOT_RECOMMENDED", reason_codes=["EXCESSIVE_PART_TO_WHOLE_CARDINALITY"], message="Too many categories make a part-to-whole chart unreadable.")
    if family in {ChartFamily.PIE, ChartFamily.DOUGHNUT} and categories > 10:
        return SuitabilityAssessment(status="SUITABLE_WITH_WARNING", reason_codes=["HIGH_PART_TO_WHOLE_CARDINALITY"], message="The chart is valid but has more categories than recommended.")
    if family == ChartFamily.BAR and categories > 20:
        return SuitabilityAssessment(status="SUITABLE_WITH_WARNING", reason_codes=["BAR_SCROLL_RECOMMENDED"], message="The result is valid; scrolling or zoom is recommended.")
    supported = {
        ChartFamily.BAR, ChartFamily.LINE, ChartFamily.AREA, ChartFamily.PIE,
        ChartFamily.DOUGHNUT, ChartFamily.SCATTER, ChartFamily.HISTOGRAM,
    }
    if family not in supported:
        return SuitabilityAssessment(status="NOT_RECOMMENDED", reason_codes=["FAMILY_REPRESENTED_NOT_AUTOMATED"], message="The family is represented by the contract but is not automatically planned yet.")
    return SuitabilityAssessment(status="SUITABLE", reason_codes=[f"{purpose.value}_COMPATIBLE"], message="The chart family matches the verified analytical result.")


__all__ = ["assess_suitability", "default_family", "purpose_for"]
