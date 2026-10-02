"""Deterministic query-intent routing for the shared Pipeline 3 NLP layer."""

from __future__ import annotations

import re
from typing import Literal

from pipelines.pipeline3_results.nlp.normalizer import (
    normalize_aggregation,
    normalize_chart,
    normalize_comparison,
    normalize_text,
)
from pipelines.pipeline3_results.nlp.schemas import IntentClassification


IntentContext = Literal["visualization", "insights"]

_VISUAL_COMMAND_RE = re.compile(r"\b(?:plot|graph|chart|visualize|visualise|visually)\b")
_SHOW_BY_RE = re.compile(r"\bshow(?:\s+me)?\b.+\bby\b")
_COMPARE_STRUCTURE_RE = re.compile(r"\bcompare\b.+\b(?:across|by|between|versus|vs|against)\b")
_VISUAL_STRUCTURE_RE = re.compile(
    r"\b(?:distribution|relationship|correlation|trend|over\s+time|versus|vs)\b"
)
_SHOW_TEMPORAL_RE = re.compile(
    r"\bshow(?:\s+me)?\b.+\b(?:daily|weekly|monthly|quarterly|yearly|annual)\b"
)
_QUESTION_START_RE = re.compile(r"^(?:which|what|how)\b")
_HOW_MANY_RE = re.compile(r"^how\s+many\b")
_COMPARATIVE_QUESTION_RE = re.compile(
    r"\b(?:highest|lowest|most|least|best|worst|top|bottom|greatest|smallest)\b"
)
_WHAT_AGGREGATE_RE = re.compile(r"^what\s+(?:is|was|are|were)\b")
_ANALYTICAL_VERB_RE = re.compile(r"\b(?:generated|performed|has|have|had)\b")
_AMBIGUOUS_RELATION_RE = re.compile(r"\b\w+\b\s+by\s+\b\w+\b")
_VAGUE_RE = re.compile(
    r"^(?:tell\s+me\s+something|analy[sz]e\s+this|show(?:\s+me)?\s+"
    r"(?:something\s+)?interesting(?:\s+stuff)?|what\s+can\s+you\s+do|"
    r"help\s+me\s+understand\s+the\s+data)$"
)


def _normalize_context(context: str | None) -> IntentContext | None:
    if context is None:
        return None
    if not isinstance(context, str):
        raise TypeError(f"context must be a string or None, got {type(context).__name__}")
    normalized = context.strip().casefold()
    if normalized not in {"visualization", "insights"}:
        raise ValueError("context must be 'visualization', 'insights', or None")
    return normalized  # type: ignore[return-value]


def classify_intent(
    raw_text: str,
    context: IntentContext | str | None = None,
) -> IntentClassification:
    """Classify request routing intent without inspecting or executing data.

    Confidence is a deterministic rule-strength label, not a calibrated
    probability: HIGH means explicit request form, MEDIUM means a clear
    structural pattern, and LOW means context fallback or insufficient signal.
    """
    if not isinstance(raw_text, str):
        raise TypeError(f"raw_text must be a string, got {type(raw_text).__name__}")
    normalized = normalize_text(raw_text)
    context_value = _normalize_context(context)

    if not normalized or _VAGUE_RE.fullmatch(normalized):
        return IntentClassification(
            intent="UNKNOWN", confidence="LOW", signals=("empty_or_vague_request",),
        )

    aggregation = normalize_aggregation(normalized)
    comparison = normalize_comparison(normalized)
    question_start = bool(_QUESTION_START_RE.search(normalized))

    # Answer-seeking analytical questions take precedence over endpoint context.
    if _HOW_MANY_RE.search(normalized):
        return IntentClassification(
            intent="INSIGHT", confidence="HIGH",
            signals=("question_word", "count_question"),
        )
    comparative_question = question_start and (
        comparison != "NONE"
        or bool(_COMPARATIVE_QUESTION_RE.search(normalized))
        or bool(_ANALYTICAL_VERB_RE.search(normalized))
    )
    aggregate_question = bool(_WHAT_AGGREGATE_RE.search(normalized)) and aggregation != "NONE"
    if comparative_question or aggregate_question:
        signals = ["question_word"]
        signals.append("comparative_question" if comparative_question else "aggregate_question")
        return IntentClassification(
            intent="INSIGHT", confidence="HIGH", signals=tuple(signals),
        )

    # Explicit display commands override an insights-context hint.
    chart = normalize_chart(normalized)
    if _VISUAL_COMMAND_RE.search(normalized):
        return IntentClassification(
            intent="VISUALIZATION", confidence="HIGH", signals=("explicit_chart_term",),
        )
    if chart is not None or _VISUAL_STRUCTURE_RE.search(normalized):
        return IntentClassification(
            intent="VISUALIZATION", confidence="HIGH", signals=("visual_structure_term",),
        )
    if _COMPARE_STRUCTURE_RE.search(normalized):
        return IntentClassification(
            intent="VISUALIZATION", confidence="MEDIUM",
            signals=("comparison_display_pattern",),
        )
    if _SHOW_BY_RE.search(normalized):
        return IntentClassification(
            intent="VISUALIZATION", confidence="MEDIUM", signals=("show_by_pattern",),
        )
    if _SHOW_TEMPORAL_RE.search(normalized):
        return IntentClassification(
            intent="VISUALIZATION", confidence="MEDIUM",
            signals=("show_temporal_pattern",),
        )

    # Context is deliberately weak. Visualization context can disambiguate a
    # concrete X-by-Y fragment; insights context cannot turn that fragment into
    # an answer-seeking question without supporting language.
    if context_value == "visualization" and _AMBIGUOUS_RELATION_RE.search(normalized):
        return IntentClassification(
            intent="VISUALIZATION", confidence="LOW",
            signals=("context_fallback", "by_relation"), context_used=True,
        )
    if context_value == "insights" and (
        aggregation != "NONE" or comparison != "NONE" or question_start
    ):
        return IntentClassification(
            intent="INSIGHT", confidence="LOW",
            signals=("context_fallback", "analytical_language"), context_used=True,
        )

    return IntentClassification(
        intent="UNKNOWN", confidence="LOW", signals=("insufficient_intent_evidence",),
    )


__all__ = ["IntentContext", "classify_intent"]
