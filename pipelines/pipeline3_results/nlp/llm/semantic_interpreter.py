"""Optional bounded LLM fallback for semantic interpretation only."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

import pandas as pd
from pydantic import ValidationError

from intelligence.llm.provider import LLMProvider
from pipelines.pipeline3_results.nlp.ambiguity_detector import detect_ambiguities
from pipelines.pipeline3_results.nlp.column_resolver import resolve_columns
from pipelines.pipeline3_results.nlp.llm.schemas import (
    LLMFollowupInterpretation,
    LLMQueryInterpretation,
)
from pipelines.pipeline3_results.nlp.normalizer import normalize_text
from pipelines.pipeline3_results.nlp.orchestrator import interpret_query
from pipelines.pipeline3_results.nlp.schemas import (
    AmbiguityDecision,
    ExtractedEntities,
    IntentClassification,
    QueryFilter,
    QueryInterpretation,
    StructuredQuery,
    TermResolution,
)
from pipelines.pipeline3_results.visualization.column_classifier import (
    _extract_column_evidence,
)
from pipelines.pipeline3_results.visualization.schemas import DatasetClassification


_CAUSAL = re.compile(r"\b(?:why|cause|caused|causing|reason for)\b", re.IGNORECASE)
_FORECAST = re.compile(
    r"\b(?:predict|forecast|will\s+.+\s+be|next\s+(?:year|month|quarter|week))\b",
    re.IGNORECASE,
)
_SEMANTIC_DESCRIPTION_FIELDS = (
    "concept_description", "semantic_description", "column_description", "description",
)


def fallback_reason(interpretation: QueryInterpretation) -> str | None:
    """Return a deterministic trigger, excluding genuine data ambiguity."""
    request_text = normalize_text(interpretation.query.raw_text)
    for decision in interpretation.ambiguity_decisions:
        if decision.status != "NEEDS_CLARIFICATION" or len(decision.choices) <= 1:
            continue
        explicitly_named = [
            choice
            for choice in decision.choices
            if re.search(
                r"\b" + re.escape(normalize_text(choice)) + r"\b",
                request_text,
            )
        ]
        # Only a single explicitly named candidate proves this is a parser
        # grammar miss rather than genuine dataset ambiguity.
        if len(explicitly_named) != 1:
            return None
    if interpretation.query.intent == "UNKNOWN":
        return "fallback:unknown_intent"
    if not interpretation.query.measure_terms and not interpretation.query.dimension_terms:
        return "fallback:no_entities"
    if interpretation.status == "INVALID":
        return "fallback:semantic_rephrase_needed"
    if interpretation.status == "NEEDS_CLARIFICATION":
        return "fallback:unsupported_grammar"
    return None


def unsupported_request_reason(raw_text: str) -> str | None:
    if _CAUSAL.search(raw_text):
        return "Causal explanation is not a supported deterministic operation."
    if _FORECAST.search(raw_text):
        return "Forecasting is not a supported deterministic operation."
    return None


def interpret_query_with_fallback(
    raw_text: str,
    dataframe: pd.DataFrame,
    *,
    pipeline1_report: Mapping[str, Any] | None = None,
    classification: DatasetClassification | None = None,
    context: str | None = None,
    provider: LLMProvider | None = None,
    deterministic: QueryInterpretation | None = None,
) -> QueryInterpretation:
    """Keep deterministic NLP primary and invoke a provider only at its boundary."""
    initial = deterministic or interpret_query(
        raw_text,
        dataframe,
        pipeline1_report=pipeline1_report,
        classification=classification,
        context=context,
    )
    if provider is None:
        return initial
    unsupported = unsupported_request_reason(raw_text)
    if unsupported is not None:
        return _controlled_failure(
            initial,
            unsupported,
            ["fallback:unsupported_capability", "llm:unsupported_request"],
        )
    reason = fallback_reason(initial)
    if reason is None:
        return initial

    active_classification = classification
    if active_classification is None:
        from pipelines.pipeline3_results.visualization.column_classifier import (
            classify_dataframe,
        )
        active_classification = classify_dataframe(
            dataframe, pipeline1_report=pipeline1_report
        )
    prompt = _query_prompt(
        raw_text, active_classification, pipeline1_report
    )
    try:
        raw = provider.generate_structured(prompt, LLMQueryInterpretation)
    except Exception as exc:
        signal = (
            "llm:invalid_output"
            if "does not satisfy" in str(exc).casefold()
            or "empty response" in str(exc).casefold()
            else "llm:provider_error"
        )
        return _controlled_failure(
            initial,
            (
                "The semantic interpreter returned an invalid structured response."
                if signal == "llm:invalid_output"
                else "The optional semantic interpreter is currently unavailable."
            ),
            [reason, signal],
        )
    try:
        interpreted = _validate_output(raw, LLMQueryInterpretation)
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError):
        return _controlled_failure(
            initial,
            "The semantic interpreter returned an invalid structured response.",
            [reason, "llm:invalid_output"],
        )
    if not interpreted.supported:
        return _controlled_failure(
            initial,
            interpreted.unsupported_reason or "The request is unsupported.",
            [reason, "llm:schema_valid", "llm:unsupported_request"],
        )
    return _resolve_llm_interpretation(
        raw_text,
        interpreted,
        dataframe,
        pipeline1_report,
        active_classification,
        reason,
    )


def interpret_followup_with_llm(
    user_text: str,
    structured_context: Mapping[str, Any],
    classification: DatasetClassification,
    *,
    pipeline1_report: Mapping[str, Any] | None = None,
    provider: LLMProvider,
) -> tuple[LLMFollowupInterpretation | None, list[str]]:
    """Interpret one modifier using structured context, never raw history."""
    unsupported = unsupported_request_reason(user_text)
    if unsupported is not None:
        return None, ["llm:unsupported_request"]
    prompt = _followup_prompt(
        user_text, structured_context, classification, pipeline1_report
    )
    try:
        raw = provider.generate_structured(prompt, LLMFollowupInterpretation)
    except Exception as exc:
        signal = (
            "llm:invalid_output"
            if "does not satisfy" in str(exc).casefold()
            or "empty response" in str(exc).casefold()
            else "llm:provider_error"
        )
        return None, [signal]
    try:
        result = _validate_output(raw, LLMFollowupInterpretation)
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError):
        return None, ["llm:invalid_output"]
    if not result.supported:
        return None, ["llm:schema_valid", "llm:unsupported_request"]
    return result, ["llm:structured_interpretation", "llm:schema_valid"]


def _validate_output(raw: Any, model_type):
    if isinstance(raw, model_type):
        return raw
    if isinstance(raw, str):
        return model_type.model_validate_json(raw)
    return model_type.model_validate(raw)


def _minimal_schema_context(
    classification: DatasetClassification,
    pipeline1_report: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    for column in classification.columns:
        evidence = _extract_column_evidence(pipeline1_report, column.name)
        description = next(
            (
                str(evidence[field]).strip()
                for field in _SEMANTIC_DESCRIPTION_FIELDS
                if evidence.get(field)
            ),
            None,
        )
        item: dict[str, Any] = {"name": column.name, "analytical_role": column.role}
        if description:
            item["semantic_description"] = description[:240]
        columns.append(item)
    return columns


def _query_prompt(
    raw_text: str,
    classification: DatasetClassification,
    pipeline1_report: Mapping[str, Any] | None,
) -> str:
    payload = {
        "request": raw_text,
        "schema_context": _minimal_schema_context(classification, pipeline1_report),
    }
    return (
        "You are a language interpreter only. Return JSON matching the supplied schema. "
        "Do not answer, calculate, generate code, SQL, formulas, chart data, or invent columns. "
        "Return user-language concepts, not guessed physical column names. Preserve ambiguity. "
        "Mark causal, forecasting, or otherwise unsupported requests as unsupported. "
        "measure_terms are numeric quantities; dimension_terms are grouping concepts; time_terms "
        "are temporal concepts. Filters are only for explicit row restrictions and must never "
        "encode aggregation or comparison. A limit is legal only with TOP_N or BOTTOM_N. "
        "A breakdown of earnings across teams uses an earnings measure and team dimension. "
        "A question asking which area brought in the most money uses a revenue measure, region "
        "dimension, SUM aggregation, and HIGHEST comparison. Include the analytical entities "
        "required by every supported interpretation.\n"
        + json.dumps(payload, ensure_ascii=True, sort_keys=True)
    )


def _followup_prompt(
    user_text: str,
    structured_context: Mapping[str, Any],
    classification: DatasetClassification,
    pipeline1_report: Mapping[str, Any] | None,
) -> str:
    payload = {
        "current_message": user_text,
        "structured_prior_context": dict(structured_context),
        "schema_context": _minimal_schema_context(classification, pipeline1_report),
    }
    return (
        "Interpret only the current follow-up as one supported structured modification. "
        "Do not answer, calculate, emit code, use raw history, or invent columns. "
        "Return JSON only and mark unsupported references as unsupported.\n"
        + json.dumps(payload, ensure_ascii=True, sort_keys=True)
    )


def _resolve_llm_interpretation(
    raw_text: str,
    llm: LLMQueryInterpretation,
    dataframe: pd.DataFrame,
    pipeline1_report: Mapping[str, Any] | None,
    classification: DatasetClassification,
    fallback_signal: str,
) -> QueryInterpretation:
    entities = ExtractedEntities(
        measure_terms=list(llm.measure_terms),
        dimension_terms=list(llm.dimension_terms),
        aggregation=llm.aggregation,
        comparison=llm.comparison,
        requested_chart=llm.requested_chart,
        filters=[
            QueryFilter(
                raw_text=f"{item.column_term} {item.operator} {item.value_term}",
                column_term=item.column_term,
                operator=item.operator,
                value_term=item.value_term,
            )
            for item in llm.filters
        ],
        time_terms=list(llm.time_terms),
        limit=llm.limit,
        signals=("llm_structured_entities",),
    )
    resolutions = resolve_columns(
        entities,
        dataframe,
        pipeline1_report=pipeline1_report,
        classification=classification,
    )
    ambiguity = detect_ambiguities(resolutions)
    measures, dimensions, times = _resolved_lists(ambiguity.decisions)
    filters = _resolved_filters(entities.filters, ambiguity.decisions)
    query_time_terms = list(llm.time_terms)
    if (
        len(query_time_terms) == 1
        and not re.search(r"\d|\b(?:last|this|next|before|after|between)\b", query_time_terms[0])
    ):
        # The adapter distinguishes a generic trend axis from an executable
        # calendar restriction. Resolution still uses the original term.
        query_time_terms = ["over time"]
    query = StructuredQuery(
        raw_text=raw_text,
        normalized_text=normalize_text(raw_text),
        intent=llm.intent,
        measure_terms=list(llm.measure_terms),
        dimension_terms=list(llm.dimension_terms),
        aggregation=llm.aggregation,
        comparison=llm.comparison,
        requested_chart=llm.requested_chart,
        filters=filters,
        time_terms=query_time_terms,
        limit=llm.limit,
        resolved_measures=measures,
        resolved_dimensions=dimensions,
        resolved_time_columns=times,
        resolutions=_attach_decisions(resolutions, ambiguity.decisions),
    )
    choices = [
        choice
        for decision in ambiguity.decisions
        if decision.status == "NEEDS_CLARIFICATION"
        for choice in decision.choices
    ]
    status = ambiguity.status if resolutions else "NEEDS_CLARIFICATION"
    signals = [
        fallback_signal,
        "llm:structured_interpretation",
        "llm:schema_valid",
        "llm:resolved_by_deterministic_engine" if status == "RESOLVED" else "llm:deterministic_resolution_incomplete",
    ]
    return QueryInterpretation(
        status=status,
        query=query,
        ambiguity_decisions=ambiguity.decisions,
        intent_confidence="LOW",
        message=ambiguity.reason if resolutions else "no_analytical_entities",
        choices=list(dict.fromkeys(choices)),
        signals=signals,
    )


def _resolved_lists(
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


def _resolved_filters(
    filters: list[QueryFilter], decisions: list[AmbiguityDecision]
) -> list[QueryFilter]:
    filter_decisions = [decision for decision in decisions if decision.kind == "FILTER"]
    return [
        item.model_copy(update={
            "resolved_column": (
                filter_decisions[index].resolved_column
                if index < len(filter_decisions) else None
            )
        })
        for index, item in enumerate(filters)
    ]


def _attach_decisions(
    resolutions: list[TermResolution], decisions: list[AmbiguityDecision]
) -> list[TermResolution]:
    return [
        resolution.model_copy(update={
            "resolved_column": decision.resolved_column,
            "reason": decision.reason,
        })
        for resolution, decision in zip(resolutions, decisions, strict=True)
    ]


def _controlled_failure(
    initial: QueryInterpretation,
    message: str,
    signals: list[str],
) -> QueryInterpretation:
    return initial.model_copy(update={
        "status": "NEEDS_CLARIFICATION",
        "message": message,
        "choices": [],
        "signals": [*initial.signals, *signals],
    })


__all__ = [
    "fallback_reason",
    "interpret_followup_with_llm",
    "interpret_query_with_fallback",
    "unsupported_request_reason",
]
