"""Compatibility adapter from shared NLP to visualization request contracts.

The adapter translates already-interpreted language into the existing
``VisualizationRequest`` model.  It performs no chart-data calculation,
aggregation, dataframe value lookup, validation, or request execution.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipelines.pipeline3_results.nlp.normalizer import normalize_text
from pipelines.pipeline3_results.nlp.schemas import FailureCode, QueryInterpretation
from pipelines.pipeline3_results.visualization.schemas import (
    DatasetClassification,
    RequestAggregation,
    VisualizationIntent,
    VisualizationRequest,
)


AdapterStatus = Literal["READY", "NEEDS_CLARIFICATION", "INVALID"]


class VisualizationAdaptation(BaseModel):
    """Internal adapter outcome; frontend responses remain service-owned."""

    model_config = ConfigDict(frozen=True)

    status: AdapterStatus
    failure_code: FailureCode | None = None
    request: VisualizationRequest | None = None
    message: str
    ambiguous_terms: list[str] = Field(default_factory=list)
    clarification_options: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)


_GENERIC_VISUAL_TERMS = {
    "chart", "graph", "plot", "visualization", "visualisation",
}


def _available_columns(classification: DatasetClassification) -> list[str]:
    return [
        column.name
        for column in classification.columns
        if column.role in {"MEASURE", "CATEGORY", "TEMPORAL", "BOOLEAN"}
    ]


def _is_generic_visual_request(interpretation: QueryInterpretation) -> bool:
    terms = [
        *interpretation.query.measure_terms,
        *interpretation.query.dimension_terms,
        *interpretation.query.time_terms,
    ]
    return bool(terms) and all(
        normalize_text(term) in _GENERIC_VISUAL_TERMS for term in terms
    )


def _clarification(
    interpretation: QueryInterpretation,
    classification: DatasetClassification,
    message: str,
    signal: str,
    *,
    choices: list[str] | None = None,
) -> VisualizationAdaptation:
    actual_choices = list(interpretation.choices if choices is None else choices)
    if not actual_choices and signal in {
        "adapter:unknown_visualization_intent",
        "adapter:generic_visualization_request",
    }:
        actual_choices = _available_columns(classification)
    ambiguous_terms = [
        decision.term
        for decision in interpretation.ambiguity_decisions
        if decision.status == "NEEDS_CLARIFICATION"
    ]
    return VisualizationAdaptation(
        status="NEEDS_CLARIFICATION",
        failure_code="AMBIGUOUS_CONCEPT" if actual_choices else None,
        message=message,
        ambiguous_terms=ambiguous_terms,
        clarification_options=actual_choices,
        signals=[*interpretation.signals, signal],
    )


def _unsupported_operation(
    interpretation: QueryInterpretation,
) -> tuple[str, str] | None:
    query = interpretation.query
    if query.comparison != "NONE":
        return (
            f"Comparison operation {query.comparison} is not supported by requested visualization execution.",
            f"adapter:unsupported_comparison:{query.comparison}",
        )
    if query.filters:
        return (
            "Filter execution is not supported by requested visualization execution.",
            "adapter:unsupported_filters",
        )
    unsupported_time = [
        term for term in query.time_terms if normalize_text(term) != "over time"
    ]
    if unsupported_time:
        return (
            "Temporal restrictions or requested granularities are not supported by requested visualization execution.",
            "adapter:unsupported_temporal_operation",
        )
    return None


def _axis_mapping(
    interpretation: QueryInterpretation,
    classification: DatasetClassification,
) -> tuple[str | None, str | None, VisualizationIntent | None, str | None]:
    query = interpretation.query
    measures = list(query.resolved_measures)
    dimensions = list(query.resolved_dimensions)
    times = list(query.resolved_time_columns)
    relationship_columns = [
        resolution.resolved_column
        for resolution in query.resolutions
        if resolution.kind == "RELATIONSHIP" and resolution.resolved_column is not None
    ]

    if relationship_columns:
        if len(relationship_columns) != 2:
            return None, None, None, "relationship_requires_exactly_two_fields"
        first, second = relationship_columns
        first_role = classification.get_column(first)
        second_role = classification.get_column(second)
        first_kind = first_role.role if first_role else None
        second_kind = second_role.role if second_role else None
        if first_kind in {"CATEGORY", "BOOLEAN", "TEMPORAL"} and second_kind == "MEASURE":
            return first, second, "RELATIONSHIP", None
        if second_kind in {"CATEGORY", "BOOLEAN", "TEMPORAL"} and first_kind == "MEASURE":
            return second, first, "RELATIONSHIP", None
        return first, second, "RELATIONSHIP", None

    if len(measures) > 1 or len(dimensions) > 1 or len(times) > 1:
        return None, None, None, "multiple_resolved_fields_require_clarification"
    if times and measures:
        return times[0], measures[0], "TREND", None
    if dimensions and measures:
        dimension_class = classification.get_column(dimensions[0])
        visual_intent: VisualizationIntent = (
            "TREND"
            if dimension_class is not None and dimension_class.role == "TEMPORAL"
            else "COMPARISON"
        )
        return dimensions[0], measures[0], visual_intent, None
    if measures:
        return measures[0], None, "DISTRIBUTION", None
    if dimensions:
        return dimensions[0], None, "DISTRIBUTION", None
    if times:
        return times[0], None, "TREND", None
    return None, None, None, "no_resolved_visualization_columns"


def adapt_query_interpretation(
    interpretation: QueryInterpretation,
    classification: DatasetClassification,
) -> VisualizationAdaptation:
    """Translate a canonical NLP interpretation into a validator-ready request."""
    if not isinstance(interpretation, QueryInterpretation):
        raise TypeError("interpretation must be a QueryInterpretation")
    if not isinstance(classification, DatasetClassification):
        raise TypeError("classification must be a DatasetClassification")

    if interpretation.status == "NEEDS_CLARIFICATION":
        return _clarification(
            interpretation,
            classification,
            interpretation.message or "The visualization request needs clarification.",
            "adapter:nlp_needs_clarification",
        )

    if interpretation.status == "INVALID":
        if interpretation.query.unsupported_chart_term is not None:
            return VisualizationAdaptation(
                status="INVALID",
                failure_code="UNSUPPORTED_CHART",
                message=(
                    f"I understood the requested dataset concepts, but "
                    f"'{interpretation.query.unsupported_chart_term}' is not a supported chart type."
                ),
                errors=[
                    "Supported chart types are BAR, LINE, SCATTER, HISTOGRAM, COUNT_BAR, and PIE."
                ],
                signals=[*interpretation.signals, "adapter:unsupported_chart"],
            )
        if _is_generic_visual_request(interpretation):
            return _clarification(
                interpretation,
                classification,
                "Specify the columns or analytical concepts to visualize.",
                "adapter:generic_visualization_request",
            )
        invalid_terms = [
            decision.term
            for decision in interpretation.ambiguity_decisions
            if decision.status == "INVALID"
        ]
        detail = ", ".join(invalid_terms) if invalid_terms else "requested concepts"
        return VisualizationAdaptation(
            status="INVALID",
            failure_code="UNRESOLVED_CONCEPT",
            message=interpretation.message or f"I couldn't find a dataset field matching '{detail}'.",
            errors=[f"No supported dataset column was found for: {detail}."],
            signals=[*interpretation.signals, "adapter:nlp_invalid"],
        )

    if interpretation.query.intent == "INSIGHT":
        return _clarification(
            interpretation,
            classification,
            "This is an insight request and is not executed by the visualization endpoint.",
            "adapter:insight_intent_not_visualized",
            choices=[],
        )

    unsupported = _unsupported_operation(interpretation)
    if unsupported is not None:
        message, signal = unsupported
        result = _clarification(
            interpretation, classification, message, signal, choices=[]
        )
        return result.model_copy(update={"failure_code": "UNSUPPORTED_OPERATION"})

    if interpretation.query.intent != "VISUALIZATION":
        result = _clarification(
            interpretation,
            classification,
            "Clarify that a visualization is requested and specify its analytical fields.",
            "adapter:unknown_visualization_intent",
        )

    x_column, y_column, visual_intent, mapping_error = _axis_mapping(
        interpretation, classification
    )
    if mapping_error is not None:
        return _clarification(
            interpretation,
            classification,
            "The resolved fields do not map unambiguously to visualization axes.",
            f"adapter:{mapping_error}",
            choices=[],
        )
        return result.model_copy(update={"failure_code": "INVALID_COMBINATION"})

    aggregation: RequestAggregation | None = (
        None if interpretation.query.aggregation == "NONE"
        else interpretation.query.aggregation
    )
    requested_columns = [
        column for column in (x_column, y_column) if column is not None
    ]
    request = VisualizationRequest(
        raw_request=interpretation.query.raw_text,
        requested_chart_type=interpretation.query.requested_chart,
        presentation_variant=interpretation.query.requested_chart_variant,
        is_explicit_chart_type=interpretation.query.requested_chart is not None,
        x_column=x_column,
        y_column=y_column,
        aggregation=aggregation,
        intent=visual_intent,
        requested_columns=requested_columns,
        parser_signals=[
            *interpretation.signals,
            "adapter:query_interpretation_mapped",
            f"adapter:x_column:{x_column}",
            f"adapter:y_column:{y_column}" if y_column else "adapter:y_column:none",
        ],
    )
    return VisualizationAdaptation(
        status="READY",
        request=request,
        message="Visualization request is ready for validation.",
        signals=list(request.parser_signals),
    )


__all__ = [
    "AdapterStatus",
    "VisualizationAdaptation",
    "adapt_query_interpretation",
]
