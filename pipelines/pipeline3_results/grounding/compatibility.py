"""Temporary bridge from canonical grounding to legacy QueryInterpretation."""

from __future__ import annotations

from pipelines.pipeline3_results.grounding.schemas import (
    AnalyticalExpression,
    GroundedConcept,
    GroundingCandidate,
    GroundingResult,
)
from pipelines.pipeline3_results.nlp.normalizer import normalize_text
from pipelines.pipeline3_results.nlp.schemas import (
    AmbiguityDecision,
    ColumnCandidate,
    QueryFilter,
    QueryInterpretation,
    StructuredQuery,
    TermResolution,
)
from pipelines.pipeline3_results.nlp.semantic_intent import SemanticIntent


_UNSUPPORTED_KINDS = {"STATISTIC", "TIME_BUCKET"}


def _expression_label(expression: AnalyticalExpression) -> str:
    if expression.kind == "COUNT_ROWS":
        return "COUNT_ROWS"
    if expression.kind == "TIME_BUCKET":
        return f"TIME_BUCKET({expression.column}, {expression.time_granularity})"
    if expression.operation:
        return f"{expression.operation}({expression.column})"
    return str(expression.column)


def _legacy_candidate(candidate: GroundingCandidate) -> ColumnCandidate:
    return ColumnCandidate(
        column_name=candidate.expression.column or _expression_label(candidate.expression),
        score=candidate.semantic_score,
        evidence_sources=list(candidate.evidence),
        semantic_evidence={
            "analytical_expression": candidate.expression.model_dump(mode="json"),
            "role_compatible": candidate.role_compatible,
            "type_compatible": candidate.type_compatible,
            "evidence_tier": candidate.evidence_tier,
        },
    )


def _resolution(concept: GroundedConcept, kind: str) -> TermResolution:
    expression = concept.resolved_expression
    return TermResolution(
        term=concept.source_concept,
        kind=kind,  # type: ignore[arg-type]
        candidates=[_legacy_candidate(item) for item in concept.candidates],
        resolved_column=expression.column if expression is not None else None,
        confidence=(
            concept.candidates[0].semantic_score if concept.candidates else None
        ),
        reason=concept.clarification_reason or (
            "resolved from DatasetKnowledge" if expression is not None else None
        ),
    )


def _decision(concept: GroundedConcept, kind: str) -> AmbiguityDecision:
    status = {
        "RESOLVED": "RESOLVED",
        "AMBIGUOUS": "NEEDS_CLARIFICATION",
        "UNRESOLVED": "INVALID",
        "UNSUPPORTED": "INVALID",
    }[concept.status]
    expression = concept.resolved_expression
    choices = [
        _expression_label(item.expression)
        for item in concept.candidates
        if item.role_compatible and item.type_compatible
    ] if concept.status == "AMBIGUOUS" else []
    return AmbiguityDecision(
        term=concept.source_concept,
        kind=kind,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        resolved_column=expression.column if expression is not None else None,
        choices=list(dict.fromkeys(choices)),
        reason=concept.clarification_reason or "resolved from DatasetKnowledge",
        signals=(f"dataset_knowledge_grounding:{concept.status.casefold()}",),
    )


def adapt_grounding_result(
    semantic: SemanticIntent,
    grounding: GroundingResult,
    raw_text: str,
) -> QueryInterpretation:
    """Convert grounded expressions without silently changing their meaning."""
    all_concepts = [
        *((item, "MEASURE") for item in grounding.measures),
        *((item, "DIMENSION") for item in grounding.dimensions),
        *((item, "FILTER") for item in grounding.filters),
        *((item, "TIME") for item in grounding.temporal),
    ]
    unsupported = [
        item for item, _kind in all_concepts
        if item.resolved_expression is not None
        and item.resolved_expression.kind in _UNSUPPORTED_KINDS
    ]
    requested_chart = "PIE" if semantic.requested_chart == "DOUGHNUT" else semantic.requested_chart
    chart_variant = "DOUGHNUT" if semantic.requested_chart == "DOUGHNUT" else None

    resolutions = [_resolution(item, kind) for item, kind in all_concepts]
    decisions = [_decision(item, kind) for item, kind in all_concepts]
    resolved_measures: list[str] = []
    resolved_dimensions: list[str] = []
    resolved_times: list[str] = []
    filters: list[QueryFilter] = []
    count_rows = False
    effective_aggregation = semantic.aggregation

    for item, kind in all_concepts:
        expression = item.resolved_expression
        if expression is None:
            continue
        if kind == "MEASURE":
            if expression.kind == "COUNT_ROWS":
                count_rows = True
                effective_aggregation = "COUNT"
            elif expression.column is not None:
                resolved_measures.append(expression.column)
                if expression.kind == "COUNT_DISTINCT":
                    effective_aggregation = "COUNT"
                elif expression.kind == "AGGREGATION" and expression.operation:
                    effective_aggregation = expression.operation  # type: ignore[assignment]
        elif kind == "DIMENSION" and expression.column is not None:
            resolved_dimensions.append(expression.column)
        elif kind == "TIME" and expression.column is not None:
            resolved_times.append(expression.column)

    for index, semantic_filter in enumerate(semantic.filters):
        grounded_filter = grounding.filters[index] if index < len(grounding.filters) else None
        expression = grounded_filter.resolved_expression if grounded_filter else None
        operator = semantic_filter.operator
        if operator not in {
            "EQUALS", "NOT_EQUALS", "GREATER_THAN", "GREATER_THAN_OR_EQUAL",
            "LESS_THAN", "LESS_THAN_OR_EQUAL", "CONTAINS", "UNKNOWN",
        }:
            operator = "UNKNOWN"
        filters.append(QueryFilter(
            raw_text=" ".join(part for part in (
                semantic_filter.field_concept,
                semantic_filter.operator,
                semantic_filter.value_concept,
            ) if part),
            column_term=semantic_filter.field_concept,
            operator=operator,  # type: ignore[arg-type]
            value_term=semantic_filter.value_concept,
            resolved_column=expression.column if expression else None,
        ))

    measure_terms = ["records"] if count_rows else list(semantic.measure_concepts)
    query_time_terms = [semantic.time_expression] if semantic.time_expression else []
    if grounding.temporal and grounding.temporal[0].resolved_expression is not None:
        if grounding.temporal[0].resolved_expression.kind == "COLUMN":
            query_time_terms = ["over time"]
    query = StructuredQuery(
        raw_text=raw_text,
        normalized_text=normalize_text(raw_text),
        intent=semantic.intent,
        measure_terms=measure_terms,
        dimension_terms=list(semantic.dimension_concepts),
        aggregation=effective_aggregation,
        comparison=semantic.comparison,
        requested_chart=requested_chart,  # type: ignore[arg-type]
        requested_chart_variant=chart_variant,  # type: ignore[arg-type]
        filters=filters,
        time_terms=query_time_terms,
        limit=semantic.limit,
        resolved_measures=list(dict.fromkeys(resolved_measures)),
        resolved_dimensions=list(dict.fromkeys(resolved_dimensions)),
        resolved_time_columns=list(dict.fromkeys(resolved_times)),
        resolutions=resolutions,
    )

    signals = [
        "semantic_interpreter:VALID_INTENT",
        "semantic_interpreter:source:LLM",
        *grounding.signals,
        "grounding:compatibility_adapter",
    ]
    choices = [choice for decision in decisions for choice in decision.choices]
    if unsupported:
        labels = ", ".join(
            _expression_label(item.resolved_expression)  # type: ignore[arg-type]
            for item in unsupported
        )
        return QueryInterpretation(
            status="NEEDS_CLARIFICATION",
            query=query,
            ambiguity_decisions=decisions,
            intent_confidence="LOW",
            message=(
                f"Grounding is correct ({labels}), but current execution does not "
                "support this analytical expression."
            ),
            signals=[
                *signals,
                "GROUNDING_CORRECT",
                "EXECUTION_UNSUPPORTED_CURRENTLY",
            ],
        )
    if grounding.status == "RESOLVED":
        status = "RESOLVED"
        message = "resolved from DatasetKnowledge"
    elif grounding.status == "AMBIGUOUS":
        status = "NEEDS_CLARIFICATION"
        message = grounding.ambiguities[0].clarification_reason
    elif grounding.status == "UNSUPPORTED":
        status = "NEEDS_CLARIFICATION"
        message = semantic.unsupported_reason or "The requested analysis is unsupported."
        signals.append("semantic_interpreter:unsupported")
    else:
        status = "INVALID"
        message = (
            grounding.unresolved[0].clarification_reason
            if grounding.unresolved else "The semantic concept could not be grounded."
        )
    return QueryInterpretation(
        status=status,  # type: ignore[arg-type]
        query=query,
        ambiguity_decisions=decisions,
        intent_confidence="LOW",
        message=message,
        choices=list(dict.fromkeys(choices)),
        signals=signals,
    )


__all__ = ["adapt_grounding_result"]
