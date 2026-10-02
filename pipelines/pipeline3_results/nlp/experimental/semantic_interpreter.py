"""Pretrained-first analytical interpretation with no dataset schema input."""

from __future__ import annotations

import json
from typing import Any, Mapping

from intelligence.llm.provider import LLMProvider
from pipelines.pipeline3_results.nlp.experimental.schemas import (
    AnalyticalInterpretation,
)


EXPERIMENTAL_SYSTEM_PROMPT = """You are an analytical-language interpreter.

Your only task is to determine what analysis the user wants. Do not answer the
question. Interpret ordinary human language semantically: different phrases
may express the same analytical meaning.

Return strict JSON matching the supplied schema. Use controlled enum values
exactly as defined by that schema. measure_concepts and dimension_concepts must
be short, normalized, domain-independent semantic phrases, never physical
dataframe column names.

Examples of semantic distinctions:
- "made the most money" means ranking by revenue.
- "sold the most" means ranking by sales volume or popularity, not revenue.
- "doing badly" has an underspecified performance metric unless conversation
  context defines it.
- "value" is a broad metric concept and may require clarification.

Preserve ambiguity. If a necessary metric or grouping concept is absent, set
requires_clarification=true and explain only what is missing. Do not invent a
metric, dimension, filter, time field, or unavailable data.

Canonical examples:
1. "Which category made us the most money?" -> intent INSIGHT,
   analytical_goal RANK_GROUPS, measure_concepts ["revenue"],
   dimension_concepts ["category"], aggregation SUM, comparison HIGHEST.
2. "Which category sells the most?" -> intent INSIGHT, analytical_goal
   RANK_GROUPS, measure_concepts ["sales volume"], dimension_concepts
   ["category"], aggregation SUM, comparison HIGHEST.
3. "Which department is doing badly?" -> dimension_concepts ["department"]
   and requires_clarification true because the performance metric is absent.
4. "Which category has the highest value?" -> measure_concepts ["value"],
   dimension_concepts ["category"], comparison HIGHEST, and
   requires_clarification true because value is semantically broad.
5. "What sells best?" -> a sales-volume/popularity ranking, but
   requires_clarification true because the entity or grouping is absent.
6. "Show me where most of the money is coming from." -> intent
   VISUALIZATION, goal PART_TO_WHOLE or GROUP_COMPARISON, revenue measure,
   and requires_clarification true because the grouping is absent.

You must never calculate results, inspect raw rows, generate Python, pandas or
SQL, choose final physical columns, or answer the analytical question. Mark
causal, predictive, or otherwise non-analytical/unsupported requests using the
unsupported fields. reasoning_summary must be a brief final rationale, not
private chain-of-thought."""


class PretrainedSemanticInterpreter:
    """Use the repository provider for one schema-constrained interpretation."""

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
        payload = {
            "user_query": user_query.strip(),
            "conversation_context": conversation_context,
        }
        return (
            EXPERIMENTAL_SYSTEM_PROMPT
            + "\n\nInterpret this request. No dataset columns or rows are available at this stage:\n"
            + json.dumps(payload, ensure_ascii=True, sort_keys=True)
        )

    def interpret(
        self,
        user_query: str,
        *,
        conversation_context: Mapping[str, Any] | str | None = None,
    ) -> AnalyticalInterpretation:
        raw = self.provider.generate_structured(
            self.build_prompt(
                user_query,
                conversation_context=conversation_context,
            ),
            AnalyticalInterpretation,
        )
        if isinstance(raw, AnalyticalInterpretation):
            return raw
        if isinstance(raw, str):
            return AnalyticalInterpretation.model_validate_json(raw)
        return AnalyticalInterpretation.model_validate(raw)


__all__ = ["EXPERIMENTAL_SYSTEM_PROMPT", "PretrainedSemanticInterpreter"]
