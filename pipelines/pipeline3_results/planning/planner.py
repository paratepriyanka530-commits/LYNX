"""Compose Phase 1 meaning and Phase 2 grounding into a typed plan."""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import json

from pipelines.pipeline3_results.grounding.schemas import (
    AnalyticalExpression,
    DatasetKnowledge,
    GroundedConcept,
    GroundingResult,
)
from pipelines.pipeline3_results.nlp.semantic_intent import SemanticIntent
from pipelines.pipeline3_results.planning.schemas import (
    AnalyticalPlan,
    GroupingExpression,
    PlanBuildResult,
    PlanFilter,
    PlanProvenance,
    SortExpression,
)
from pipelines.pipeline3_results.planning.validator import validate_analytical_plan


_FILTER_OPERATORS = {
    "EQUALS", "NOT_EQUALS", "GREATER_THAN", "GREATER_THAN_OR_EQUAL",
    "LESS_THAN", "LESS_THAN_OR_EQUAL", "BETWEEN", "BEFORE", "AFTER",
}


def _resolved(items: list[GroundedConcept]) -> list[AnalyticalExpression]:
    return [
        item.resolved_expression
        for item in items
        if item.resolved_expression is not None
    ]


def _expression_label(expression: AnalyticalExpression) -> str:
    if expression.kind == "COUNT_ROWS":
        return "COUNT_ROWS"
    if expression.kind == "TIME_BUCKET":
        return f"TIME_BUCKET({expression.column}, {expression.time_granularity})"
    if expression.operation:
        return f"{expression.operation}({expression.column})"
    return str(expression.column)


def _operation(semantic: SemanticIntent, measures: list[AnalyticalExpression]) -> str | None:
    if semantic.analytical_goal == "EXPLORATORY":
        return "EXPLORATORY"
    if semantic.comparison == "TOP_N" or semantic.analytical_goal == "TOP_N":
        return "TOP_N"
    if semantic.comparison == "BOTTOM_N" or semantic.analytical_goal == "BOTTOM_N":
        return "BOTTOM_N"
    if semantic.comparison in {"HIGHEST", "LOWEST"} or semantic.analytical_goal == "RANK_GROUPS":
        return "RANK"
    if any(item.kind == "STATISTIC" for item in measures):
        return "STATISTIC"
    return {
        "SIMPLE_AGGREGATION": "AGGREGATE",
        "GROUP_COMPARISON": "GROUP_AGGREGATE",
        "TREND": "TREND",
        "DISTRIBUTION": "DISTRIBUTION",
        "RELATIONSHIP": "RELATIONSHIP",
        "COUNT_ENTITIES": "COUNT",
        "PART_TO_WHOLE": "PART_TO_WHOLE",
        "EXPLORATORY": "EXPLORATORY",
    }.get(semantic.analytical_goal)


def _typed_value(raw: str, *, temporal: bool, numeric: bool) -> tuple[str, object]:
    clean = raw.strip()
    if temporal:
        try:
            if "T" in clean or " " in clean:
                return "DATETIME", datetime.fromisoformat(clean)
            return "DATE", date.fromisoformat(clean)
        except ValueError:
            return "STRING", clean
    if numeric:
        try:
            value = float(clean)
            return "NUMBER", int(value) if value.is_integer() else value
        except ValueError:
            return "STRING", clean
    if clean.casefold() in {"true", "false"}:
        return "BOOLEAN", clean.casefold() == "true"
    return "STRING", clean


def _plan_id(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"plan_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _clarification_options(grounding: GroundingResult) -> list[str]:
    return list(dict.fromkeys(
        _expression_label(candidate.expression)
        for concept in grounding.ambiguities
        for candidate in concept.candidates
        if candidate.role_compatible and candidate.type_compatible
    ))


def _provenance(
    semantic: SemanticIntent,
    grounding: GroundingResult,
    *,
    decisions: list[str] | None = None,
    warnings: list[str] | None = None,
) -> PlanProvenance:
    resolved = [
        expression
        for concept in [
            *grounding.measures,
            *grounding.dimensions,
            *grounding.temporal,
            *grounding.filters,
        ]
        if (expression := concept.resolved_expression) is not None
    ]
    return PlanProvenance(
        source_intent=semantic.model_dump(mode="json", exclude={"requested_chart"}),
        grounding_status=grounding.status,
        resolved_expressions=[item.model_dump(mode="json") for item in resolved],
        planner_decisions=decisions or [],
        warnings=warnings or [],
    )


def build_analytical_plan(
    semantic_intent: SemanticIntent,
    grounding_result: GroundingResult,
    dataset_knowledge: DatasetKnowledge,
) -> PlanBuildResult:
    """Build and validate one plan without inspecting rows or executing work."""
    if not isinstance(semantic_intent, SemanticIntent):
        raise TypeError("semantic_intent must be a SemanticIntent")
    if not isinstance(grounding_result, GroundingResult):
        raise TypeError("grounding_result must be a GroundingResult")
    if not isinstance(dataset_knowledge, DatasetKnowledge):
        raise TypeError("dataset_knowledge must be DatasetKnowledge")

    base_provenance = _provenance(semantic_intent, grounding_result)
    if grounding_result.status == "AMBIGUOUS":
        return PlanBuildResult(
            status="NEEDS_CLARIFICATION",
            message=(grounding_result.ambiguities[0].clarification_reason or "Grounding is ambiguous."),
            clarification_options=_clarification_options(grounding_result),
            provenance=base_provenance,
        )
    if grounding_result.status == "UNRESOLVED":
        return PlanBuildResult(
            status="NEEDS_CLARIFICATION",
            message=(grounding_result.unresolved[0].clarification_reason or "A required concept is missing or unresolved."),
            provenance=base_provenance,
        )
    if grounding_result.status == "UNSUPPORTED" or semantic_intent.unsupported:
        return PlanBuildResult(
            status="UNSUPPORTED",
            message=semantic_intent.unsupported_reason or "The requested analysis is unsupported.",
            provenance=base_provenance,
        )

    measures = _resolved(grounding_result.measures)
    operation = _operation(semantic_intent, measures)
    if operation is None:
        return PlanBuildResult(
            status="UNSUPPORTED",
            message=f"Analytical goal {semantic_intent.analytical_goal} is not representable.",
            provenance=base_provenance,
        )

    dimensions = [
        GroupingExpression(kind="GROUP_BY", column=expression.column)
        for expression in _resolved(grounding_result.dimensions)
        if expression.column is not None
    ]
    temporal: list[GroupingExpression] = []
    for expression in _resolved(grounding_result.temporal):
        if expression.column is None:
            continue
        temporal.append(GroupingExpression(
            kind="TIME_BUCKET" if expression.kind == "TIME_BUCKET" else "GROUP_BY",
            column=expression.column,
            time_granularity=expression.time_granularity,
        ))

    filters: list[PlanFilter] = []
    for index, semantic_filter in enumerate(semantic_intent.filters):
        if semantic_filter.operator not in _FILTER_OPERATORS:
            return PlanBuildResult(
                status="UNSUPPORTED",
                message=f"Filter operator {semantic_filter.operator} is not representable safely.",
                provenance=base_provenance,
            )
        if semantic_filter.value_concept is None:
            return PlanBuildResult(
                status="NEEDS_CLARIFICATION",
                message=f"Filter {semantic_filter.field_concept} requires a value.",
                provenance=base_provenance,
            )
        grounded = grounding_result.filters[index].resolved_expression
        if grounded is None or grounded.column is None:
            return PlanBuildResult(
                status="NEEDS_CLARIFICATION",
                message=f"Filter field {semantic_filter.field_concept} is unresolved.",
                provenance=base_provenance,
            )
        column = dataset_knowledge.get_column(grounded.column)
        if column is None:
            return PlanBuildResult(status="INVALID", message="Grounding referenced an unknown filter column.", provenance=base_provenance)
        from pipelines.pipeline3_results.grounding.engine import (
            is_numeric_eligible,
            is_temporal_eligible,
        )
        value_type, value = _typed_value(
            semantic_filter.value_concept,
            temporal=is_temporal_eligible(column),
            numeric=is_numeric_eligible(column),
        )
        if semantic_filter.operator == "BETWEEN":
            return PlanBuildResult(
                status="NEEDS_CLARIFICATION",
                message="BETWEEN requires two separately grounded boundary values.",
                provenance=base_provenance,
            )
        filters.append(PlanFilter(
            column=grounded.column,
            operator=semantic_filter.operator,  # type: ignore[arg-type]
            value_type=value_type,  # type: ignore[arg-type]
            value=value,  # type: ignore[arg-type]
        ))

    sort: list[SortExpression] = []
    if operation in {"RANK", "TOP_N", "BOTTOM_N"}:
        direction = "ASC" if operation == "BOTTOM_N" or semantic_intent.comparison == "LOWEST" else "DESC"
        sort.append(SortExpression(target="MEASURE", direction=direction, measure_index=0))
    elif operation == "TREND" and temporal:
        sort.append(SortExpression(
            target="TEMPORAL", direction="ASC", column=temporal[0].column
        ))

    decisions = [
        f"operation:{operation}",
        *[f"measure:{_expression_label(item)}" for item in measures],
        *[f"dimension:{item.kind}({item.column})" for item in dimensions],
        *[f"temporal:{item.kind}({item.column})" for item in temporal],
    ]
    provenance = _provenance(
        semantic_intent, grounding_result, decisions=decisions
    )
    payload = {
        "operation": operation,
        "measures": [item.model_dump(mode="json") for item in measures],
        "dimensions": [item.model_dump(mode="json") for item in dimensions],
        "filters": [item.model_dump(mode="json") for item in filters],
        "temporal": [item.model_dump(mode="json") for item in temporal],
        "sort": [item.model_dump(mode="json") for item in sort],
        "limit": semantic_intent.limit,
        "comparison": semantic_intent.comparison,
        "provenance": provenance.model_dump(mode="json"),
    }
    plan = AnalyticalPlan(plan_id=_plan_id(payload), **payload)  # type: ignore[arg-type]
    validation = validate_analytical_plan(plan, dataset_knowledge)
    if validation.status != "READY":
        return PlanBuildResult(
            status="INVALID",
            message=validation.message,
            errors=list(validation.errors),
            provenance=provenance,
        )
    return PlanBuildResult(
        status="READY", plan=plan, message="Analytical plan is ready.",
        provenance=provenance,
    )


__all__ = ["build_analytical_plan"]
