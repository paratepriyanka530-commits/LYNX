"""Translate canonical shared-NLP interpretations into controlled insight queries."""

from __future__ import annotations

from pipelines.pipeline3_results.insights.schemas import (
    CountMode,
    InsightAdaptation,
    InsightQuery,
    InsightQueryType,
)
from pipelines.pipeline3_results.nlp.normalizer import normalize_text
from pipelines.pipeline3_results.nlp.schemas import Aggregation, QueryInterpretation
from pipelines.pipeline3_results.visualization.schemas import DatasetClassification


_ROW_COUNT_TERMS = {
    "record", "records", "row", "rows", "observation", "observations",
    "entry", "entries",
}
_RELATIONSHIP_WORDS = {"relationship", "related", "correlation", "correlated", "association"}


def _controlled(
    interpretation: QueryInterpretation,
    status: str,
    message: str,
    signal: str,
    *,
    errors: list[str] | None = None,
    failure_code: str | None = None,
) -> InsightAdaptation:
    return InsightAdaptation(
        status=status,  # type: ignore[arg-type]
        failure_code=failure_code,  # type: ignore[arg-type]
        message=message,
        clarification_options=list(interpretation.choices),
        errors=errors or [],
        signals=[*interpretation.signals, signal],
    )


def _is_row_count_request(interpretation: QueryInterpretation) -> bool:
    query = interpretation.query
    if query.aggregation != "COUNT" or query.dimension_terms:
        return False
    normalized_terms = {
        normalize_text(term) for term in query.measure_terms
    }
    return bool(normalized_terms) and normalized_terms.issubset(_ROW_COUNT_TERMS)


def _contains_relationship_language(raw_text: str) -> bool:
    return bool(set(normalize_text(raw_text).split()).intersection(_RELATIONSHIP_WORDS))


def adapt_insight_query(
    interpretation: QueryInterpretation,
    classification: DatasetClassification,
) -> InsightAdaptation:
    """Create an executable InsightQuery or a controlled non-execution result."""
    if not isinstance(interpretation, QueryInterpretation):
        raise TypeError("interpretation must be a QueryInterpretation")
    if not isinstance(classification, DatasetClassification):
        raise TypeError("classification must be a DatasetClassification")

    row_count_request = _is_row_count_request(interpretation)

    if interpretation.query.intent == "VISUALIZATION":
        return _controlled(
            interpretation,
            "NEEDS_CLARIFICATION",
            "This is a visualization request and is not executed by Insights.",
            "insight_adapter:visualization_intent_not_executed",
        )
    if interpretation.query.intent == "UNKNOWN":
        reason = (
            "Relationship analysis is not supported in the initial Insights engine."
            if _contains_relationship_language(interpretation.query.raw_text)
            else "Clarify the factual question to answer from the dataset."
        )
        signal = (
            "insight_adapter:unsupported_relationship"
            if _contains_relationship_language(interpretation.query.raw_text)
            else "insight_adapter:unknown_intent"
        )
        return _controlled(
            interpretation,
            "NEEDS_CLARIFICATION",
            reason,
            signal,
            failure_code=(
                "UNSUPPORTED_ANALYSIS"
                if _contains_relationship_language(interpretation.query.raw_text)
                else None
            ),
        )

    if interpretation.status == "NEEDS_CLARIFICATION":
        return _controlled(
            interpretation,
            "NEEDS_CLARIFICATION",
            interpretation.message or "The insight question needs clarification.",
            "insight_adapter:nlp_needs_clarification",
            failure_code="AMBIGUOUS_CONCEPT",
        )
    if interpretation.status == "INVALID" and not row_count_request:
        invalid_terms = [
            decision.term
            for decision in interpretation.ambiguity_decisions
            if decision.status == "INVALID"
        ]
        detail = ", ".join(invalid_terms) if invalid_terms else "requested concepts"
        return _controlled(
            interpretation,
            "INVALID",
            interpretation.message or f"I couldn't find a dataset field matching '{detail}'.",
            "insight_adapter:nlp_invalid",
            errors=[f"No supported dataset column was found for: {detail}."],
            failure_code="UNRESOLVED_CONCEPT",
        )

    query = interpretation.query
    if query.filters:
        return _controlled(
            interpretation,
            "NEEDS_CLARIFICATION",
            "Filter execution is not supported by the initial Insights engine.",
            "insight_adapter:unsupported_filters",
            failure_code="UNSUPPORTED_OPERATION",
        )
    if query.time_terms or query.resolved_time_columns:
        return _controlled(
            interpretation,
            "NEEDS_CLARIFICATION",
            "Temporal restrictions and time-grouped insight queries are not yet supported.",
            "insight_adapter:unsupported_temporal_query",
            failure_code="UNSUPPORTED_OPERATION",
        )
    if any(resolution.kind == "RELATIONSHIP" for resolution in query.resolutions):
        return _controlled(
            interpretation,
            "NEEDS_CLARIFICATION",
            "Relationship analysis is not supported in the initial Insights engine.",
            "insight_adapter:unsupported_relationship",
            failure_code="UNSUPPORTED_ANALYSIS",
        )
    if len(query.resolved_measures) > 1 or len(query.resolved_dimensions) > 1:
        return _controlled(
            interpretation,
            "NEEDS_CLARIFICATION",
            "Multi-measure or multi-dimension insight queries are not yet supported.",
            "insight_adapter:unsupported_multiple_fields",
            failure_code="UNSUPPORTED_ANALYSIS",
        )

    measure = query.resolved_measures[0] if query.resolved_measures else None
    dimension = query.resolved_dimensions[0] if query.resolved_dimensions else None
    aggregation: Aggregation = query.aggregation
    comparison = query.comparison
    count_mode: CountMode | None = None

    if row_count_request:
        measure = None
        aggregation = "COUNT"
        count_mode = "COUNT_ROWS"
    elif aggregation == "COUNT":
        if measure is None and dimension is not None and comparison != "NONE":
            # Natural popularity questions ("which category is ordered most")
            # count rows per resolved group; the dimension is not also treated
            # as a duplicated measure column.
            count_mode = "COUNT_ROWS"
        else:
            count_column = measure or dimension
            if count_column is not None:
                column = classification.get_column(count_column)
                count_mode = (
                    "COUNT_DISTINCT"
                    if column is not None and column.role == "IDENTIFIER"
                    else "COUNT_NON_NULL"
                )
                measure = count_column

    if aggregation == "NONE" and comparison != "NONE" and measure is not None:
        column = classification.get_column(measure)
        if column is not None and column.role == "IDENTIFIER":
            aggregation = "COUNT"
            count_mode = "COUNT_DISTINCT"
        else:
            # A grouped numeric measure without an explicit aggregation uses
            # SUM, matching the controlled "products by revenue" contract.
            aggregation = "SUM"

    query_type: InsightQueryType
    if comparison == "TOP_N":
        query_type = "TOP_N"
    elif comparison == "BOTTOM_N":
        query_type = "BOTTOM_N"
    elif comparison in {"HIGHEST", "LOWEST"}:
        query_type = "GROUPED_COMPARISON"
    else:
        query_type = "SIMPLE_AGGREGATION"

    insight_query = InsightQuery(
        original_text=query.raw_text,
        query_type=query_type,
        measure_column=measure,
        dimension_column=dimension,
        aggregation=aggregation,
        comparison=comparison,
        limit=query.limit,
        count_mode=count_mode,
        filters=list(query.filters),
        temporal_columns=list(query.resolved_time_columns),
        signals=[
            *interpretation.signals,
            "insight_adapter:query_mapped",
            f"insight_adapter:query_type:{query_type}",
        ],
    )
    return InsightAdaptation(
        status="READY",
        query=insight_query,
        message="Insight query is ready for validation.",
        signals=list(insight_query.signals),
    )


__all__ = ["adapt_insight_query"]
