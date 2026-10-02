"""Qwen-first language understanding with provenance-only deterministic checks."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from intelligence.llm.provider import LLMProvider
from pipelines.pipeline3_results.nlp.experimental.schemas_v2 import (
    AnalyticalInterpretationV2,
    SemanticField,
    SourceType,
)


V2_SYSTEM_PROMPT = """You are LYNX's analytical-language interpreter.

Your only job is to faithfully represent what analysis the user means. Do not
answer the question, inspect data, choose dataframe columns, calculate, execute,
or recommend a chart. Return strict JSON matching the supplied schema.

PRIORITIES, IN ORDER:
1. Preserve what the user explicitly said.
2. Infer only meaning linguistically entailed by the user's words.
3. Never invent a missing metric, entity, dimension, numeric limit, chart type,
   filter, time range, or comparison.
4. Represent missing required information in missing_slots.
5. Represent a present but broad/multi-meaning concept in ambiguities; ambiguity
   is not the same as missing.
6. Ask one precise clarification question only when the requested operation
   requires the missing or ambiguous information.

Every SemanticField has value, source, and evidence:
- EXPLICIT: the value was directly stated. Copy the shortest exact supporting
  phrase from the query into evidence.
- INFERRED: the value is strongly entailed. Copy the shortest exact phrase that
  entails it into evidence.
- MISSING: set both value and evidence to null.

missing_slots is only for analytically REQUIRED absent information whose absence
prevents the requested operation. Never list optional absent limit, chart,
filter, or time fields there. ambiguities is only for a concept the user
actually supplied; never speculate about an absent chart, metric, or dimension.

Hard constraints:
- limit is non-null only for an explicitly stated number. "most", "best", and
  "worst" are comparisons, not limit=1. "top products" has no numeric limit.
- requested_chart is non-null only when the user names a chart type. Words such
  as "show", "compare", "trend", and "over time" do not name a chart.
- filters are present only when the user explicitly restricts rows.
- do not blindly use SUM. Use MEAN for explicit average, COUNT for explicit
  how-many/frequency meaning, and NONE when aggregation is not entailed.
- exploratory requests such as "what stands out?" are valid EXPLORATORY goals;
  absent measures and dimensions are not required missing slots for that goal.
- measure and dimension values are free semantic phrases, never physical column
  names or domain-specific codes.
- interpretation_summary is a short final rationale, not chain-of-thought.

GOOD INFERENCE versus BAD INVENTION:

User: "Which category made us the most money?"
GOOD: revenue is INFERRED from "made us the most money"; category is EXPLICIT;
SUM is INFERRED; HIGHEST is INFERRED; limit and chart are MISSING.

User: "What sells the most?"
BAD: dimension product, limit 1.
GOOD: measure ["sales volume"] INFERRED from "sells"; HIGHEST INFERRED from
"most"; dimensions MISSING; limit MISSING; clarification asks what entity or
group should be ranked.

User: "Compare price and quantity"
BAD: BOTTOM_N, BAR, or a winner.
GOOD: goal RELATIONSHIP; measures ["price", "quantity"] EXPLICIT; comparison
NONE INFERRED from "compare"; dimensions, limit, and chart MISSING; no
clarification is required.

User: "Which month was strongest?"
BAD: inventing a measure named strength.
GOOD: dimension ["month"] EXPLICIT; HIGHEST INFERRED from "strongest"; measure
MISSING; clarification asks which metric defines strongest.

User: "Which category has the highest value?"
GOOD: measure ["value"] EXPLICIT and an ambiguity entry explains that value is
broad; it is not replaced with revenue.

User: "How did energy usage change throughout the year?"
GOOD: TREND; measure ["energy usage"] EXPLICIT; time expression year EXPLICIT;
requested_chart MISSING.

User: "What stands out?"
GOOD: INSIGHT and EXPLORATORY; measures and dimensions MISSING; no
clarification, because exploratory analysis does not require specified fields.

Exact structural exemplar for "Which category made us the most money?":
{"intent":{"value":"INSIGHT","source":"INFERRED","evidence":"Which category made us the most money"},"analytical_goal":{"value":"RANK_GROUPS","source":"INFERRED","evidence":"most money"},"measures":{"value":["revenue"],"source":"INFERRED","evidence":"made us the most money"},"dimensions":{"value":["category"],"source":"EXPLICIT","evidence":"category"},"aggregation":{"value":"SUM","source":"INFERRED","evidence":"made us the most money"},"comparison":{"value":"HIGHEST","source":"INFERRED","evidence":"most"},"limit":{"value":null,"source":"MISSING","evidence":null},"requested_chart":{"value":null,"source":"MISSING","evidence":null},"filters":{"value":null,"source":"MISSING","evidence":null},"time_expression":{"value":null,"source":"MISSING","evidence":null},"ambiguities":[],"missing_slots":[],"requires_clarification":false,"clarification_question":null,"unsupported":false,"unsupported_reason":null,"interpretation_summary":"Ranks categories by inferred revenue; no chart or limit was requested."}
"""


class StructuralInterpretationError(ValueError):
    """The model output is typed but contradicts its own provenance claims."""

    def __init__(self, message: str, result: AnalyticalInterpretationV2):
        super().__init__(message)
        self.result = result


_NUMBER_WORDS: dict[int, str] = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen", 20: "twenty",
}
_CHART_TERMS: dict[str, tuple[str, ...]] = {
    "BAR": ("bar chart", "bar graph", "bar plot"),
    "LINE": ("line chart", "line graph", "line plot"),
    "SCATTER": ("scatter chart", "scatter graph", "scatter plot"),
    "HISTOGRAM": ("histogram",),
    "COUNT_BAR": ("count bar", "count chart", "count plot"),
    "PIE": ("pie chart", "pie graph", "pie plot"),
    "DOUGHNUT": ("doughnut chart", "doughnut graph", "doughnut plot",
                  "donut chart", "donut graph", "donut plot"),
}


def _semantic_fields(
    result: AnalyticalInterpretationV2,
) -> list[tuple[str, SemanticField[Any]]]:
    return [
        ("intent", result.intent),
        ("analytical_goal", result.analytical_goal),
        ("measure", result.measures),
        ("dimension", result.dimensions),
        ("aggregation", result.aggregation),
        ("comparison", result.comparison),
        ("limit", result.limit),
        ("requested_chart", result.requested_chart),
        ("filter", result.filters),
        ("time_expression", result.time_expression),
    ]


def validate_interpretation_structure(
    query: str,
    result: AnalyticalInterpretationV2,
) -> None:
    """Validate provenance and contradictions without interpreting language."""
    query_folded = " ".join(query.casefold().split())
    for name, field in _semantic_fields(result):
        if field.source == SourceType.MISSING:
            continue
        evidence = " ".join((field.evidence or "").casefold().split())
        if not evidence or evidence not in query_folded:
            raise StructuralInterpretationError(
                f"{name} evidence must be an exact phrase from the user query",
                result,
            )

    if result.requested_chart.value is not None:
        evidence = (result.requested_chart.evidence or "").casefold()
        permitted = _CHART_TERMS[result.requested_chart.value]
        if not any(term in evidence for term in permitted):
            raise StructuralInterpretationError(
                "requested_chart evidence does not explicitly name that chart",
                result,
            )

    if result.limit.value is not None:
        evidence = (result.limit.evidence or "").casefold()
        value = result.limit.value
        word = _NUMBER_WORDS.get(value)
        numeric_match = re.search(rf"(?<!\d){value}(?!\d)", evidence)
        if not numeric_match and (word is None or not re.search(rf"\b{word}\b", evidence)):
            raise StructuralInterpretationError(
                "limit evidence does not contain the stated numeric limit",
                result,
            )

    fields_by_slot = dict(_semantic_fields(result))
    for missing in result.missing_slots:
        field = fields_by_slot[missing.slot]
        if field.source != SourceType.MISSING:
            raise StructuralInterpretationError(
                f"missing slot {missing.slot!r} is not marked MISSING",
                result,
            )


class PretrainedSemanticInterpreterV2:
    def __init__(self, provider: LLMProvider):
        if not isinstance(provider, LLMProvider):
            raise TypeError("provider must implement LLMProvider")
        self.provider = provider

    def build_prompt(
        self,
        user_query: str,
        *,
        conversation_context: Mapping[str, Any] | str | None = None,
    ) -> str:
        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError("user_query must be a non-empty string")
        return V2_SYSTEM_PROMPT + "\n\nREQUEST:\n" + json.dumps(
            {
                "user_query": user_query.strip(),
                "conversation_context": conversation_context,
            },
            ensure_ascii=True,
            sort_keys=True,
        )

    def interpret(
        self,
        user_query: str,
        *,
        conversation_context: Mapping[str, Any] | str | None = None,
    ) -> AnalyticalInterpretationV2:
        raw = self.provider.generate_structured(
            self.build_prompt(user_query, conversation_context=conversation_context),
            AnalyticalInterpretationV2,
        )
        if isinstance(raw, AnalyticalInterpretationV2):
            result = raw
        elif isinstance(raw, str):
            result = AnalyticalInterpretationV2.model_validate_json(raw)
        else:
            result = AnalyticalInterpretationV2.model_validate(raw)
        validate_interpretation_structure(user_query, result)
        return result


__all__ = [
    "PretrainedSemanticInterpreterV2",
    "StructuralInterpretationError",
    "V2_SYSTEM_PROMPT",
    "validate_interpretation_structure",
]
