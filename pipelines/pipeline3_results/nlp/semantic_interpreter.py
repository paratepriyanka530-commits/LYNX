"""Provider-first production semantic interpretation for Pipeline 3."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Mapping

from pydantic import ValidationError

from intelligence.llm.provider import (
    LLMProvider,
    LLMProviderError,
    LLMProviderTimeoutError,
    LLMProviderUnavailableError,
)
from pipelines.pipeline3_results.nlp.semantic_intent import (
    SemanticIntent,
    SemanticInterpretationResult,
)

if TYPE_CHECKING:
    import pandas as pd
    from pipelines.pipeline3_results.nlp.schemas import QueryInterpretation
    from pipelines.pipeline3_results.visualization.schemas import DatasetClassification


SEMANTIC_INTENT_SYSTEM_PROMPT = """You are LYNX's analytical-language interpreter.
Return strict JSON matching SemanticIntent. Represent meaning only; do not answer,
calculate, select dataframe columns, inspect data, or generate code.

Priorities:
1. Preserve explicitly stated concepts as short, free semantic phrases.
2. Infer only meaning that the user's words justify.
3. Never invent a metric, dimension, chart, numeric limit, filter, or time range.
4. Put required absent information in missing_slots. Put a supplied but broad or
   multi-meaning concept in ambiguities. Never mark one slot as both.
5. Set needs_clarification and one precise question only when missing or ambiguous
   information blocks the request.
6. Set requested_chart only when the user explicitly names a chart type. Set limit
   only when the user explicitly states a number; most/best/least/worst are not 1.
7. Use SUM for explicit totals, MEAN for explicit averages, COUNT for explicit
   how-many/frequency meaning, and NONE when aggregation is not justified.
8. Exploratory requests such as "What stands out?" need no invented fields and no
   clarification. Trend language does not itself request a LINE chart.
9. Mark causal, predictive, or otherwise unsupported requests explicitly.

Examples:
- "What sells the most?": INSIGHT, RANK_GROUPS, measure ["sales volume"],
  comparison HIGHEST, missing dimension, no limit, no chart.
- "Compare price and quantity": RELATIONSHIP, measures ["price", "quantity"],
  comparison NONE, no chart.
- "Which month was strongest?": dimension ["month"], comparison HIGHEST,
  missing measure; do not invent a strength metric.
- "How did energy usage change throughout the year?": TREND, measure
  ["energy usage"], time_expression "throughout the year", no chart.
- "Which category has the highest value?": preserve measure ["value"] and mark
  that supplied measure ambiguous; do not replace it with revenue.
"""


class ProviderInterpretationError(ValueError):
    """The provider returned typed output that violates query-level structure."""


_NUMBER_WORDS = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty",
)
_CHART_TERMS: dict[str, tuple[str, ...]] = {
    "BAR": ("bar chart", "bar graph", "bar plot"),
    "LINE": ("line chart", "line graph", "line plot"),
    "SCATTER": ("scatter chart", "scatter graph", "scatter plot"),
    "HISTOGRAM": ("histogram",),
    "COUNT_BAR": ("count bar", "count chart", "count plot"),
    "PIE": ("pie chart", "pie graph", "pie plot"),
    "DOUGHNUT": (
        "doughnut chart", "doughnut graph", "doughnut plot",
        "donut chart", "donut graph", "donut plot", "doughnut", "donut",
    ),
}


def validate_semantic_intent_structure(query: str, intent: SemanticIntent) -> None:
    """Check query-level invariants without re-interpreting analytical meaning."""
    folded = " ".join(query.casefold().split())
    if intent.limit is not None:
        word = _NUMBER_WORDS[intent.limit - 1] if 1 <= intent.limit <= 20 else None
        if not re.search(rf"(?<!\d){intent.limit}(?!\d)", folded) and not (
            word and re.search(rf"\b{word}\b", folded)
        ):
            raise ProviderInterpretationError(
                "numeric limit is not explicitly present in the user query"
            )
    if intent.requested_chart is not None:
        terms = _CHART_TERMS[intent.requested_chart]
        if not any(term in folded for term in terms):
            raise ProviderInterpretationError(
                "requested chart is not explicitly present in the user query"
            )


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _failure_state(exc: BaseException) -> str:
    chain = _exception_chain(exc)
    if any(isinstance(item, (LLMProviderTimeoutError, TimeoutError)) for item in chain):
        return "PROVIDER_TIMEOUT"
    if any(isinstance(item, (LLMProviderUnavailableError, ConnectionError)) for item in chain):
        return "PROVIDER_UNAVAILABLE"
    if isinstance(exc, (ValidationError, ProviderInterpretationError, ValueError, TypeError, json.JSONDecodeError)):
        return "INVALID_MODEL_OUTPUT"
    if isinstance(exc, LLMProviderError):
        return "PROVIDER_ERROR"
    return "PROVIDER_ERROR"


class SemanticInterpreter:
    """Call one model-agnostic provider and return an explicit typed outcome."""

    def __init__(self, provider: LLMProvider):
        if not isinstance(provider, LLMProvider):
            raise TypeError("provider must implement LLMProvider")
        self.provider = provider

    def build_prompt(
        self,
        user_text: str,
        *,
        conversation_context: Mapping[str, Any] | None = None,
    ) -> str:
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be a non-empty string")
        payload = {
            "user_text": user_text.strip(),
            "conversation_context": dict(conversation_context or {}),
        }
        return (
            SEMANTIC_INTENT_SYSTEM_PROMPT
            + "\n\nInterpret this request:\n"
            + json.dumps(payload, ensure_ascii=True, sort_keys=True)
        )

    def interpret(
        self,
        user_text: str,
        *,
        conversation_context: Mapping[str, Any] | None = None,
    ) -> SemanticInterpretationResult:
        prompt = self.build_prompt(
            user_text, conversation_context=conversation_context
        )
        try:
            raw = self.provider.generate_structured(prompt, SemanticIntent)
            if isinstance(raw, SemanticIntent):
                intent = raw
            elif isinstance(raw, str):
                intent = SemanticIntent.model_validate_json(raw)
            else:
                intent = SemanticIntent.model_validate(raw)
            validate_semantic_intent_structure(user_text, intent)
        except Exception as exc:
            state = _failure_state(exc)
            messages = {
                "INVALID_MODEL_OUTPUT": "The semantic provider returned invalid structured output.",
                "PROVIDER_UNAVAILABLE": "The semantic provider is unavailable.",
                "PROVIDER_TIMEOUT": "The semantic provider timed out.",
                "PROVIDER_ERROR": "The semantic provider failed.",
            }
            return SemanticInterpretationResult(
                state=state,  # type: ignore[arg-type]
                source="LLM",
                message=messages[state],
            )
        return SemanticInterpretationResult(
            state="VALID_INTENT",
            source="LLM",
            semantic_intent=intent,
        )


def interpret_query_pretrained_first(
    user_text: str,
    dataframe: "pd.DataFrame",
    *,
    pipeline1_report: Mapping[str, Any] | None = None,
    classification: "DatasetClassification | None" = None,
    context: str | None = None,
    conversation_context: Mapping[str, Any] | None = None,
    provider: LLMProvider | None = None,
    deterministic_fallback: bool = True,
) -> "QueryInterpretation":
    """Use the provider first and expose any deterministic fallback in signals."""
    from pipelines.pipeline3_results.nlp.compatibility_adapter import (
        adapt_semantic_intent,
    )
    from pipelines.pipeline3_results.nlp.orchestrator import interpret_query

    if provider is not None:
        result = SemanticInterpreter(provider).interpret(
            user_text, conversation_context=conversation_context
        )
        if result.state == "VALID_INTENT":
            assert result.semantic_intent is not None
            return adapt_semantic_intent(
                result.semantic_intent,
                user_text,
                dataframe,
                pipeline1_report=pipeline1_report,
                classification=classification,
            )
        if not deterministic_fallback:
            fallback = interpret_query(
                user_text,
                dataframe,
                pipeline1_report=pipeline1_report,
                classification=classification,
                context=context,
            )
            return fallback.model_copy(update={
                "status": "NEEDS_CLARIFICATION",
                "message": result.message,
                "signals": [
                    *fallback.signals,
                    f"semantic_interpreter:{result.state}",
                    "semantic_interpreter:source:LLM",
                    "semantic_interpreter:fallback_disabled",
                ],
            })
        failure_state = result.state
    else:
        failure_state = "PROVIDER_UNAVAILABLE"

    fallback = interpret_query(
        user_text,
        dataframe,
        pipeline1_report=pipeline1_report,
        classification=classification,
        context=context,
    )
    return fallback.model_copy(update={
        "signals": [
            *fallback.signals,
            f"semantic_interpreter:{failure_state}",
            "semantic_interpreter:source:DETERMINISTIC_FALLBACK",
        ],
    })


__all__ = [
    "ProviderInterpretationError",
    "SEMANTIC_INTENT_SYSTEM_PROMPT",
    "SemanticInterpreter",
    "interpret_query_pretrained_first",
    "validate_semantic_intent_structure",
]
