"""Focused contract tests for the production pretrained-first semantic layer."""

from __future__ import annotations

import json
from typing import Any, Type

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from intelligence.llm.provider import (
    LLMProvider,
    LLMProviderTimeoutError,
    LLMProviderUnavailableError,
)
from pipelines.pipeline3_results.insights.results_service import build_insight_results
from pipelines.pipeline3_results.nlp.compatibility_adapter import adapt_semantic_intent
from pipelines.pipeline3_results.nlp.semantic_intent import SemanticIntent
from pipelines.pipeline3_results.nlp.semantic_interpreter import SemanticInterpreter


class RecordingProvider(LLMProvider):
    def __init__(self, response: Any = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = 0
        self.prompts: list[str] = []
        self.response_models: list[Type[BaseModel]] = []

    def generate_structured(self, prompt: str, response_model: Type[BaseModel]):
        self.calls += 1
        self.prompts.append(prompt)
        self.response_models.append(response_model)
        if self.error is not None:
            raise self.error
        return self.response


def _intent(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "intent": "INSIGHT",
        "analytical_goal": "SIMPLE_AGGREGATION",
        "measure_concepts": ["revenue"],
        "dimension_concepts": [],
        "aggregation": "SUM",
        "comparison": "NONE",
        "limit": None,
        "requested_chart": None,
        "filters": [],
        "time_expression": None,
        "missing_slots": [],
        "ambiguities": [],
        "needs_clarification": False,
        "clarification_question": None,
        "unsupported": False,
        "unsupported_reason": None,
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    ("query", "updates", "checks"),
    [
        (
            "Which region made the most revenue?",
            dict(analytical_goal="RANK_GROUPS", dimension_concepts=["region"],
                 comparison="HIGHEST"),
            {"analytical_goal": "RANK_GROUPS", "comparison": "HIGHEST"},
        ),
        (
            "What sells the most?",
            dict(analytical_goal="RANK_GROUPS", measure_concepts=["sales volume"],
                 aggregation="NONE", comparison="HIGHEST",
                 missing_slots=[{"slot": "dimension", "reason": "The ranked entity is missing."}],
                 needs_clarification=True,
                 clarification_question="Which entity or group should be ranked?"),
            {"limit": None, "dimension_concepts": []},
        ),
        (
            "Which department is doing badly?",
            dict(analytical_goal="RANK_GROUPS", measure_concepts=[],
                 dimension_concepts=["department"], aggregation="NONE",
                 comparison="LOWEST",
                 missing_slots=[{"slot": "measure", "reason": "The performance metric is missing."}],
                 needs_clarification=True,
                 clarification_question="Which metric defines doing badly?"),
            {"measure_concepts": []},
        ),
        (
            "Which category has the highest value?",
            dict(analytical_goal="RANK_GROUPS", measure_concepts=["value"],
                 dimension_concepts=["category"], aggregation="NONE",
                 comparison="HIGHEST",
                 ambiguities=[{"slot": "measure", "concept": "value",
                               "reason": "Value is semantically broad."}],
                 needs_clarification=True,
                 clarification_question="Which value metric should be used?"),
            {"measure_concepts": ["value"]},
        ),
        (
            "What stands out?",
            dict(analytical_goal="EXPLORATORY", measure_concepts=[],
                 aggregation="NONE"),
            {"needs_clarification": False, "analytical_goal": "EXPLORATORY"},
        ),
        (
            "Compare price and quantity",
            dict(analytical_goal="RELATIONSHIP", measure_concepts=["price", "quantity"],
                 aggregation="NONE"),
            {"comparison": "NONE", "requested_chart": None},
        ),
        (
            "Make a doughnut chart of revenue by category",
            dict(intent="VISUALIZATION", analytical_goal="PART_TO_WHOLE",
                 dimension_concepts=["category"], requested_chart="DOUGHNUT"),
            {"requested_chart": "DOUGHNUT"},
        ),
        (
            "How did energy usage change throughout the year?",
            dict(analytical_goal="TREND", measure_concepts=["energy usage"],
                 aggregation="NONE", time_expression="throughout the year"),
            {"requested_chart": None, "time_expression": "throughout the year"},
        ),
        (
            "Show the top 5 products by revenue",
            dict(analytical_goal="TOP_N", dimension_concepts=["products"],
                 comparison="TOP_N", limit=5),
            {"limit": 5},
        ),
        (
            "Which attack type occurred most frequently?",
            dict(analytical_goal="RANK_GROUPS", measure_concepts=["attack frequency"],
                 dimension_concepts=["attack type"], aggregation="COUNT",
                 comparison="HIGHEST"),
            {"limit": None},
        ),
    ],
)
def test_cross_domain_semantic_contract(query, updates, checks) -> None:
    result = SemanticInterpreter(RecordingProvider(_intent(**updates))).interpret(query)
    assert result.state == "VALID_INTENT"
    assert result.semantic_intent is not None
    for field, expected in checks.items():
        assert getattr(result.semantic_intent, field) == expected


def test_no_invented_chart_or_dimension() -> None:
    provider = RecordingProvider(_intent(
        analytical_goal="TREND", measure_concepts=["energy usage"],
        aggregation="NONE", time_expression="throughout the year",
    ))
    result = SemanticInterpreter(provider).interpret(
        "How did energy usage change throughout the year?"
    )
    assert result.semantic_intent is not None
    assert result.semantic_intent.requested_chart is None
    assert result.semantic_intent.dimension_concepts == []


def test_missing_and_ambiguity_conflict_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SemanticIntent.model_validate(_intent(
            measure_concepts=[],
            missing_slots=[{"slot": "measure", "reason": "Missing."}],
            ambiguities=[{"slot": "measure", "concept": "value", "reason": "Broad."}],
            needs_clarification=True,
            clarification_question="Which measure?",
        ))


def test_malformed_provider_json() -> None:
    result = SemanticInterpreter(RecordingProvider("{not-json")).interpret("Total revenue")
    assert result.state == "INVALID_MODEL_OUTPUT"
    assert result.semantic_intent is None


def test_provider_timeout() -> None:
    result = SemanticInterpreter(RecordingProvider(
        error=LLMProviderTimeoutError("secret timeout detail")
    )).interpret("Total revenue")
    assert result.state == "PROVIDER_TIMEOUT"
    assert "secret" not in (result.message or "")


def test_provider_unavailable() -> None:
    result = SemanticInterpreter(RecordingProvider(
        error=LLMProviderUnavailableError("secret endpoint")
    )).interpret("Total revenue")
    assert result.state == "PROVIDER_UNAVAILABLE"


def test_extra_unknown_field_is_rejected() -> None:
    result = SemanticInterpreter(RecordingProvider(
        _intent(final_answer="The answer is 42")
    )).interpret("Total revenue")
    assert result.state == "INVALID_MODEL_OUTPUT"


def test_unsupported_request_is_distinct_from_provider_failure() -> None:
    result = SemanticInterpreter(RecordingProvider(_intent(
        analytical_goal="UNKNOWN", measure_concepts=[], aggregation="NONE",
        unsupported=True,
        unsupported_reason="Forecasting is not supported by this analytical path.",
    ))).interpret("Predict next year's revenue")
    assert result.state == "VALID_INTENT"
    assert result.semantic_intent is not None
    assert result.semantic_intent.unsupported is True


def test_minimal_structured_conversation_context_is_sent() -> None:
    provider = RecordingProvider(_intent(measure_concepts=["salary"], aggregation="MEAN"))
    SemanticInterpreter(provider).interpret(
        "Use average salary instead",
        conversation_context={
            "previous_intent": "INSIGHT",
            "previous_measure_concepts": ["revenue"],
            "pending_clarification_slot": None,
        },
    )
    prompt = provider.prompts[0]
    assert "previous_measure_concepts" in prompt
    assert "revenue" in prompt
    request_payload = prompt.split("Interpret this request:\n", 1)[1]
    assert "dataframe" not in request_payload.casefold()


def test_compatibility_adapter_uses_existing_grounding() -> None:
    dataframe = pd.DataFrame({
        "Revenue": [10.0, 20.0, 30.0],
        "Region": ["N", "S", "N"],
    })
    semantic = SemanticIntent.model_validate(_intent(
        analytical_goal="RANK_GROUPS",
        dimension_concepts=["Region"],
        comparison="HIGHEST",
    ))
    adapted = adapt_semantic_intent(semantic, "Which Region has highest Revenue?", dataframe)
    assert adapted.status == "RESOLVED"
    assert adapted.query.resolved_measures == ["Revenue"]
    assert adapted.query.resolved_dimensions == ["Region"]
    assert "semantic_interpreter:source:LLM" in adapted.signals


def test_production_service_calls_provider_before_deterministic_parser() -> None:
    dataframe = pd.DataFrame({"revenue": [10.0, 20.0, 30.0]})
    provider = RecordingProvider(_intent())
    result = build_insight_results(dataframe, "total revenue", llm_provider=provider)
    assert provider.calls == 1
    assert result["status"] == "READY"
    assert "semantic_interpreter:source:LLM" in result["signals"]


def test_deterministic_fallback_source_and_failure_state_are_exposed() -> None:
    dataframe = pd.DataFrame({"revenue": [10.0, 20.0, 30.0]})
    provider = RecordingProvider(error=LLMProviderUnavailableError("offline"))
    result = build_insight_results(dataframe, "total revenue", llm_provider=provider)
    assert result["status"] == "READY"
    assert "semantic_interpreter:PROVIDER_UNAVAILABLE" in result["signals"]
    assert "semantic_interpreter:source:DETERMINISTIC_FALLBACK" in result["signals"]


def test_language_fixture_is_model_independent() -> None:
    path = __file__.replace("test_semantic_intent.py", "fixtures/semantic_intent_corpus.json")
    with open(path, encoding="utf-8") as handle:
        corpus = json.load(handle)
    assert len(corpus) >= 10
    assert {item["domain"] for item in corpus} >= {
        "retail", "hr", "healthcare", "banking", "iot", "cybersecurity"
    }
    assert all("query" in item and "expected" in item for item in corpus)
