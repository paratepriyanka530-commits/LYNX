"""Deterministic structural validation for canonical analytical plans."""

from __future__ import annotations

from pipelines.pipeline3_results.grounding.engine import (
    is_category_eligible,
    is_identifier_eligible,
    is_numeric_eligible,
    is_temporal_eligible,
)
from pipelines.pipeline3_results.grounding.schemas import DatasetKnowledge
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan, PlanValidation


_NUMERIC_OPERATIONS = {"SUM", "MEAN", "MEDIAN", "MIN", "MAX", "STDDEV", "RANGE"}
_RANK_OPERATIONS = {"RANK", "TOP_N", "BOTTOM_N"}
_COMPUTED_MEASURE_KINDS = {
    "AGGREGATION", "COUNT_ROWS", "COUNT_NON_NULL", "COUNT_DISTINCT", "STATISTIC",
}


def validate_analytical_plan(
    plan: AnalyticalPlan,
    dataset_knowledge: DatasetKnowledge,
) -> PlanValidation:
    if not isinstance(plan, AnalyticalPlan):
        raise TypeError("plan must be an AnalyticalPlan")
    if not isinstance(dataset_knowledge, DatasetKnowledge):
        raise TypeError("dataset_knowledge must be DatasetKnowledge")

    errors: list[str] = []
    for expression in plan.measures:
        column = dataset_knowledge.get_column(expression.column) if expression.column else None
        if expression.column and column is None:
            errors.append(f"Unknown measure column: {expression.column}.")
            continue
        if expression.operation in _NUMERIC_OPERATIONS and column is not None and not is_numeric_eligible(column):
            errors.append(f"{expression.operation} requires a numeric measure: {expression.column}.")
        if expression.kind == "COUNT_DISTINCT" and column is not None and not is_identifier_eligible(column):
            errors.append(f"COUNT_DISTINCT requires an identifier/entity source: {expression.column}.")

    for grouping in plan.dimensions:
        column = dataset_knowledge.get_column(grouping.column)
        if column is None:
            errors.append(f"Unknown dimension column: {grouping.column}.")
        elif not (is_category_eligible(column) or is_temporal_eligible(column) or is_identifier_eligible(column)):
            errors.append(f"Column is not grouping-compatible: {grouping.column}.")

    for grouping in plan.temporal:
        column = dataset_knowledge.get_column(grouping.column)
        if column is None:
            errors.append(f"Unknown temporal column: {grouping.column}.")
        elif not is_temporal_eligible(column):
            errors.append(f"Temporal grouping requires a temporal source: {grouping.column}.")

    for item in plan.filters:
        column = dataset_knowledge.get_column(item.column)
        if column is None:
            errors.append(f"Unknown filter column: {item.column}.")
            continue
        if item.operator in {"GREATER_THAN", "GREATER_THAN_OR_EQUAL", "LESS_THAN", "LESS_THAN_OR_EQUAL", "BETWEEN"}:
            if not (is_numeric_eligible(column) or is_temporal_eligible(column)):
                errors.append(f"Ordered filter requires numeric or temporal source: {item.column}.")
        if item.operator in {"BEFORE", "AFTER"} and not is_temporal_eligible(column):
            errors.append(f"{item.operator} requires a temporal source: {item.column}.")
        if item.value_type == "NUMBER" and not is_numeric_eligible(column):
            errors.append(f"Numeric filter value is incompatible with: {item.column}.")
        if item.value_type in {"DATE", "DATETIME"} and not is_temporal_eligible(column):
            errors.append(f"Temporal filter value is incompatible with: {item.column}.")
        if item.operator in {"GREATER_THAN", "GREATER_THAN_OR_EQUAL", "LESS_THAN", "LESS_THAN_OR_EQUAL", "BETWEEN"}:
            if is_numeric_eligible(column) and item.value_type != "NUMBER":
                errors.append(f"Numeric ordered filter requires numeric values: {item.column}.")
            if is_temporal_eligible(column) and item.value_type not in {"DATE", "DATETIME"}:
                errors.append(f"Temporal ordered filter requires typed temporal values: {item.column}.")

    for item in plan.sort:
        if item.target == "MEASURE" and (
            item.measure_index is None or item.measure_index >= len(plan.measures)
        ):
            errors.append("Measure sort index is outside the plan's measures.")
        if item.target == "DIMENSION" and item.column not in {group.column for group in plan.dimensions}:
            errors.append(f"Sort column is not a plan dimension: {item.column}.")
        if item.target == "TEMPORAL" and item.column not in {group.column for group in plan.temporal}:
            errors.append(f"Sort column is not a plan temporal grouping: {item.column}.")

    if plan.operation in _RANK_OPERATIONS:
        if not plan.measures:
            errors.append(f"{plan.operation} requires a rankable measure.")
        if not plan.dimensions:
            errors.append(f"{plan.operation} requires a grouping dimension.")
        if not plan.sort or plan.sort[0].target != "MEASURE":
            errors.append(f"{plan.operation} requires explicit measure sorting.")
        if plan.measures and plan.measures[0].kind not in _COMPUTED_MEASURE_KINDS:
            errors.append(f"{plan.operation} requires a computed rankable measure.")
    if plan.operation == "AGGREGATE" and (
        len(plan.measures) != 1 or plan.measures[0].kind not in _COMPUTED_MEASURE_KINDS
    ):
        errors.append("AGGREGATE requires one computed measure expression.")
    if plan.operation == "GROUP_AGGREGATE" and (
        len(plan.measures) != 1
        or plan.measures[0].kind not in _COMPUTED_MEASURE_KINDS
        or not plan.dimensions
    ):
        errors.append("GROUP_AGGREGATE requires one computed measure and a dimension.")
    if plan.operation in {"TOP_N", "BOTTOM_N"} and plan.limit is None:
        errors.append(f"{plan.operation} requires an explicit positive limit.")
    if plan.operation == "RELATIONSHIP" and len(plan.measures) != 2:
        errors.append("RELATIONSHIP requires exactly two measures.")
    elif plan.operation == "RELATIONSHIP" and any(item.kind != "COLUMN" for item in plan.measures):
        errors.append("RELATIONSHIP requires unaggregated column measures.")
    if plan.operation == "TREND" and not plan.temporal:
        errors.append("TREND requires temporal grouping.")
    if plan.operation == "DISTRIBUTION" and (
        len(plan.measures) != 1 or plan.measures[0].kind != "COLUMN"
    ):
        errors.append("DISTRIBUTION requires one unaggregated column measure.")
    if plan.operation == "STATISTIC" and (
        len(plan.measures) != 1 or plan.measures[0].kind != "STATISTIC"
    ):
        errors.append("STATISTIC requires one statistical measure expression.")
    if plan.operation == "COUNT" and (
        len(plan.measures) != 1 or any(
            item.kind not in {"COUNT_ROWS", "COUNT_NON_NULL", "COUNT_DISTINCT"}
            for item in plan.measures
        )
    ):
        errors.append("COUNT requires explicit row, non-null, or distinct count semantics.")
    if plan.operation == "PART_TO_WHOLE":
        if len(plan.measures) != 1 or not plan.dimensions:
            errors.append("PART_TO_WHOLE requires one measure and a grouping dimension.")
        elif plan.measures[0].operation not in {"SUM", "COUNT", "COUNT_DISTINCT"} and plan.measures[0].kind != "COUNT_ROWS":
            errors.append("PART_TO_WHOLE requires an additive or count measure.")
    if plan.operation == "EXPLORATORY" and (plan.measures or plan.dimensions or plan.temporal):
        errors.append("EXPLORATORY must not fabricate analytical fields.")

    if errors:
        return PlanValidation(
            status="INVALID",
            message="The analytical plan failed structural validation.",
            errors=errors,
        )
    return PlanValidation(
        status="READY",
        plan=plan,
        message="The analytical plan is structurally valid.",
    )


__all__ = ["validate_analytical_plan"]
