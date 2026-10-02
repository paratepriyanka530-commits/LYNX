"""Canonical deterministic orchestration for shared Pipeline 3 language NLP.

This module composes normalization, intent classification, entity extraction,
column resolution, and ambiguity detection.  It performs no dataframe value
lookups, calculations, chart generation, LLM calls, or runtime integration.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipelines.pipeline3_results.nlp.ambiguity_detector import detect_ambiguities
from pipelines.pipeline3_results.nlp.column_resolver import resolve_columns
from pipelines.pipeline3_results.nlp.entity_extractor import extract_entities
from pipelines.pipeline3_results.nlp.intent_classifier import (
    IntentContext,
    classify_intent,
)
from pipelines.pipeline3_results.nlp.normalizer import normalize_text
from pipelines.pipeline3_results.nlp.schemas import (
    AmbiguityDecision,
    QueryFilter,
    QueryInterpretation,
    StructuredQuery,
    TermResolution,
)
from pipelines.pipeline3_results.visualization.schemas import DatasetClassification


def _append_signal(signals: list[str], signal: str) -> None:
    if signal not in signals:
        signals.append(signal)


def _resolved_filters(
    filters: list[QueryFilter],
    decisions: list[AmbiguityDecision],
) -> list[QueryFilter]:
    """Attach resolved field names while preserving operators and value terms."""
    filter_decisions: dict[str, list[AmbiguityDecision]] = {}
    for decision in decisions:
        if decision.kind == "FILTER":
            filter_decisions.setdefault(normalize_text(decision.term), []).append(decision)

    consumed: dict[str, int] = {}
    resolved: list[QueryFilter] = []
    for query_filter in filters:
        if not query_filter.column_term:
            resolved.append(query_filter)
            continue
        key = normalize_text(query_filter.column_term)
        position = consumed.get(key, 0)
        choices = filter_decisions.get(key, [])
        decision = choices[position] if position < len(choices) else None
        consumed[key] = position + 1
        resolved.append(query_filter.model_copy(update={
            "resolved_column": decision.resolved_column if decision else None,
        }))
    return resolved


def _resolved_term_lists(
    decisions: list[AmbiguityDecision],
) -> tuple[list[str], list[str], list[str]]:
    measures: list[str] = []
    dimensions: list[str] = []
    times: list[str] = []
    for decision in decisions:
        if decision.status != "RESOLVED" or decision.resolved_column is None:
            continue
        if decision.kind in {"MEASURE", "RELATIONSHIP"}:
            measures.append(decision.resolved_column)
        elif decision.kind == "DIMENSION":
            dimensions.append(decision.resolved_column)
        elif decision.kind == "TIME":
            times.append(decision.resolved_column)
    return measures, dimensions, times


def _decision_resolutions(
    resolutions: list[TermResolution],
    decisions: list[AmbiguityDecision],
) -> list[TermResolution]:
    """Preserve candidates while attaching Phase 5 outcomes to query terms."""
    return [
        resolution.model_copy(update={
            "resolved_column": decision.resolved_column,
            "reason": decision.reason,
        })
        for resolution, decision in zip(resolutions, decisions, strict=True)
    ]


def interpret_query(
    raw_text: str,
    dataframe: pd.DataFrame,
    pipeline1_report: Any = None,
    classification: DatasetClassification | None = None,
    context: IntentContext | str | None = None,
) -> QueryInterpretation:
    """Run the five approved deterministic NLP stages and assemble one result."""
    if not isinstance(raw_text, str):
        raise TypeError(f"raw_text must be a string, got {type(raw_text).__name__}")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"dataframe must be a pandas DataFrame, got {type(dataframe).__name__}")

    normalized = normalize_text(raw_text)
    intent = classify_intent(raw_text, context=context)
    entities = extract_entities(raw_text, intent=intent)
    resolutions = resolve_columns(
        entities,
        dataframe,
        pipeline1_report=pipeline1_report,
        classification=classification,
    )
    ambiguity = detect_ambiguities(resolutions)

    resolved_measures, resolved_dimensions, resolved_times = _resolved_term_lists(
        ambiguity.decisions
    )
    query = StructuredQuery(
        raw_text=raw_text,
        normalized_text=normalized,
        intent=intent.intent,
        measure_terms=list(entities.measure_terms),
        dimension_terms=list(entities.dimension_terms),
        aggregation=entities.aggregation,
        comparison=entities.comparison,
        requested_chart=entities.requested_chart,
        requested_chart_variant=entities.requested_chart_variant,
        unsupported_chart_term=entities.unsupported_chart_term,
        filters=_resolved_filters(entities.filters, ambiguity.decisions),
        time_terms=list(entities.time_terms),
        limit=entities.limit,
        resolved_measures=resolved_measures,
        resolved_dimensions=resolved_dimensions,
        resolved_time_columns=resolved_times,
        resolutions=_decision_resolutions(resolutions, ambiguity.decisions),
        confidence=entities.confidence,
    )

    signals: list[str] = []
    _append_signal(
        signals,
        "normalize:text_normalized" if normalized else "normalize:empty_text",
    )
    for signal in intent.signals:
        _append_signal(signals, f"intent:{signal}")
    for signal in entities.signals:
        _append_signal(signals, f"extract:{signal}")
    for resolution in resolutions:
        if not resolution.candidates:
            _append_signal(signals, "resolver:no_supported_candidate")
        for candidate in resolution.candidates:
            for source in candidate.evidence_sources:
                _append_signal(signals, f"resolver:{source}")
    for decision in ambiguity.decisions:
        for signal in decision.signals:
            _append_signal(signals, f"ambiguity:{signal}")
    for signal in ambiguity.signals:
        _append_signal(signals, f"ambiguity:{signal}")

    choices: list[str] = []
    for decision in ambiguity.decisions:
        if decision.status == "NEEDS_CLARIFICATION":
            for choice in decision.choices:
                if choice not in choices:
                    choices.append(choice)

    # An empty interpretation needs clarification regardless of the vacuous
    # all-resolved result for an empty decision list.  Otherwise Phase 5's
    # INVALID > clarification > resolved precedence is authoritative.
    ambiguous = [d for d in ambiguity.decisions if d.status == "NEEDS_CLARIFICATION"]
    unresolved = [d for d in ambiguity.decisions if d.status == "INVALID"]

    if entities.unsupported_chart_term is not None:
        status = "INVALID"
        message = f"Chart type '{entities.unsupported_chart_term}' is not supported."
        _append_signal(signals, "failure:UNSUPPORTED_CHART")
        _append_signal(signals, "orchestrator:invalid")
    elif not resolutions:
        status = "NEEDS_CLARIFICATION"
        message = "empty_request" if not normalized else "no_analytical_entities"
        _append_signal(signals, "orchestrator:no_analytical_entities")
        _append_signal(signals, "orchestrator:needs_clarification")
    else:
        status = ambiguity.status
        if ambiguous:
            decision = ambiguous[0]
            readable = ", ".join(choice.replace("_", " ").title() for choice in decision.choices)
            message = f"'{decision.term}' could refer to {readable}. Which one should I use?"
            _append_signal(signals, "failure:AMBIGUOUS_CONCEPT")
        elif unresolved:
            decision = unresolved[0]
            message = f"I couldn't find a dataset field matching '{decision.term}'."
            _append_signal(signals, "failure:UNRESOLVED_CONCEPT")
        else:
            message = ambiguity.reason
        _append_signal(signals, f"orchestrator:{status.casefold()}")

    return QueryInterpretation(
        status=status,
        query=query,
        ambiguity_decisions=ambiguity.decisions,
        intent_confidence=intent.confidence,
        context_used=intent.context_used,
        message=message,
        choices=choices,
        signals=signals,
        confidence=entities.confidence,
    )


orchestrate_query = interpret_query


__all__ = ["interpret_query", "orchestrate_query"]
