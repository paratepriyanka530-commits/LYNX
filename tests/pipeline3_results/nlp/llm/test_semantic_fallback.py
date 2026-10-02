"""Security and behavior tests for the optional semantic LLM fallback."""

from __future__ import annotations

import json
from typing import Any, Type

import pandas as pd
import pytest
from pydantic import BaseModel

from intelligence.llm.provider import LLMProvider
from pipelines.pipeline3_results.conversation.orchestrator import ConversationOrchestrator
from pipelines.pipeline3_results.insights.results_service import build_insight_results
from pipelines.pipeline3_results.nlp.llm.schemas import LLMQueryInterpretation
from pipelines.pipeline3_results.visualization.results_service import (
    build_visualization_results,
)


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


@pytest.fixture
def dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "salary": [50.0, 70.0, 60.0, 80.0, 90.0, 100.0],
        "department": ["A", "A", "B", "B", "C", "C"],
        "sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "revenue": [12.0, 22.0, 32.0, 42.0, 52.0, 62.0],
        "region": ["North", "North", "South", "South", "East", "East"],
        "product": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "month": pd.to_datetime([
            "2025-01-01", "2025-02-01", "2025-03-01",
            "2025-04-01", "2025-05-01", "2025-06-01",
        ]),
    })


def _query(**updates):
    payload = {
        "supported": True,
        "unsupported_reason": None,
        "intent": "VISUALIZATION",
        "measure_terms": ["salary"],
        "dimension_terms": ["department"],
        "aggregation": "MEAN",
        "comparison": "NONE",
        "requested_chart": None,
        "filters": [],
        "time_terms": [],
        "limit": None,
    }
    payload.update(updates)
    return payload


def test_deterministic_query_does_not_call_llm(dataframe) -> None:
    provider = RecordingProvider(_query())
    result = build_visualization_results(
        dataframe, "show average salary by department", llm_provider=provider
    )
    assert result["status"] == "READY"
    assert provider.calls == 0


def test_unknown_phrasing_invokes_fallback_and_normal_resolution(dataframe) -> None:
    provider = RecordingProvider(_query())
    result = build_visualization_results(
        dataframe,
        "give me a breakdown of what people earn across each department",
        llm_provider=provider,
    )
    assert provider.calls == 1
    assert result["status"] == "READY"
    assert result["visualizations"][0]["x_column"] == "department"
    assert result["visualizations"][0]["y_column"] == "salary"
    assert "llm:resolved_by_deterministic_engine" in result["signals"]


def test_insight_rephrase_fallback(dataframe) -> None:
    provider = RecordingProvider(_query(
        intent="INSIGHT", measure_terms=["sales"], dimension_terms=["region"],
        aggregation="SUM", comparison="HIGHEST",
    ))
    result = build_insight_results(
        dataframe, "Which area brought in the most money?", llm_provider=provider
    )
    # "money" is now a deterministic analytical concept, but both sales and
    # revenue are equally supported in this schema.  Genuine ambiguity must
    # remain a clarification and must not be guessed by the LLM.
    assert provider.calls == 0
    assert result["status"] == "NEEDS_CLARIFICATION"
    assert result["failure_code"] == "AMBIGUOUS_CONCEPT"
    assert set(result["clarification_options"]) == {"sales", "revenue"}


def test_trend_rephrase_fallback(dataframe) -> None:
    provider = RecordingProvider(_query(
        measure_terms=["revenue"], dimension_terms=[], aggregation="SUM",
        requested_chart="LINE", time_terms=["month"],
    ))
    result = build_visualization_results(
        dataframe,
        "I want to see how revenue changed as months passed",
        llm_provider=provider,
    )
    assert result["status"] == "READY"
    assert result["visualizations"][0]["chart_type"] == "LINE"
    assert result["visualizations"][0]["x_column"] == "month"


def test_malformed_llm_json_is_safe(dataframe) -> None:
    provider = RecordingProvider("{not-json")
    result = build_visualization_results(
        dataframe, "give me a breakdown of earnings", llm_provider=provider
    )
    assert result["status"] == "NEEDS_CLARIFICATION"
    assert "llm:invalid_output" in result["signals"]


def test_invalid_enum_is_rejected(dataframe) -> None:
    provider = RecordingProvider(_query(aggregation="AVERAGEISH"))
    result = build_visualization_results(
        dataframe, "give me a breakdown of earnings", llm_provider=provider
    )
    assert result["status"] == "NEEDS_CLARIFICATION"
    assert "llm:invalid_output" in result["signals"]


def test_unknown_output_fields_are_rejected(dataframe) -> None:
    provider = RecordingProvider(_query(python_code="df.mean()"))
    result = build_visualization_results(
        dataframe, "give me a breakdown of earnings", llm_provider=provider
    )
    assert result["status"] == "NEEDS_CLARIFICATION"
    assert "llm:invalid_output" in result["signals"]


def test_executable_semantic_term_is_rejected(dataframe) -> None:
    provider = RecordingProvider(_query(measure_terms=["import os; exec(x)"]))
    result = build_visualization_results(
        dataframe, "give me a breakdown of earnings", llm_provider=provider
    )
    assert result["status"] == "NEEDS_CLARIFICATION"
    assert "llm:invalid_output" in result["signals"]


def test_invented_column_is_rejected_by_deterministic_resolver(dataframe) -> None:
    provider = RecordingProvider(_query(measure_terms=["imaginary profit"]))
    result = build_visualization_results(
        dataframe, "give me a breakdown of business performance", llm_provider=provider
    )
    assert provider.calls == 1
    assert result["status"] == "INVALID"
    assert "llm:deterministic_resolution_incomplete" in result["signals"]


def test_llm_cannot_override_measure_ambiguity() -> None:
    dataframe = pd.DataFrame({
        "gross_sales": [10.0, 20.0], "net_sales": [8.0, 16.0],
        "region": ["N", "S"],
    })
    provider = RecordingProvider(_query(measure_terms=["gross sales"]))
    result = build_visualization_results(
        dataframe, "show sales by region", llm_provider=provider
    )
    assert provider.calls == 0
    assert result["status"] == "NEEDS_CLARIFICATION"
    assert result["clarification_options"] == ["gross_sales", "net_sales"]


def test_llm_cannot_override_temporal_ambiguity() -> None:
    dataframe = pd.DataFrame({
        "sales": [10.0, 20.0, 30.0, 40.0],
        "order_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
        "created_at": pd.to_datetime(["2024-01-02", "2024-02-02", "2024-03-02", "2024-04-02"]),
    })
    provider = RecordingProvider(_query(time_terms=["order date"], requested_chart="LINE"))
    result = build_visualization_results(
        dataframe, "show sales over time", llm_provider=provider
    )
    assert provider.calls == 0
    assert result["status"] == "NEEDS_CLARIFICATION"
    assert set(result["clarification_options"]) == {"order_date", "created_at"}


@pytest.mark.parametrize(
    "question",
    ["why did revenue drop?", "what will sales be next year?"],
)
def test_unsupported_capability_is_not_sent_to_provider(dataframe, question) -> None:
    provider = RecordingProvider(_query())
    result = build_insight_results(dataframe, question, llm_provider=provider)
    assert provider.calls == 0
    assert result["status"] == "NEEDS_CLARIFICATION"
    assert "llm:unsupported_request" in result["signals"]


def test_prompt_contains_schema_only_and_no_row_values() -> None:
    dataframe = pd.DataFrame({
        "opaque_measure": [987654321.123, 123456789.987],
        "opaque_group": ["SECRET_ROW_ALPHA", "SECRET_ROW_BETA"],
    })
    provider = RecordingProvider(_query(
        measure_terms=["opaque measure"], dimension_terms=["opaque group"]
    ))
    build_visualization_results(
        dataframe, "give me an unfamiliar breakdown", llm_provider=provider
    )
    prompt = provider.prompts[0]
    assert "opaque_measure" in prompt and "analytical_role" in prompt
    assert "SECRET_ROW_ALPHA" not in prompt
    assert "987654321.123" not in prompt
    assert "schema_context" in prompt


def test_provider_failure_is_controlled(dataframe) -> None:
    provider = RecordingProvider(error=RuntimeError("offline"))
    result = build_visualization_results(
        dataframe, "give me a breakdown of earnings", llm_provider=provider
    )
    assert result["status"] == "NEEDS_CLARIFICATION"
    assert "llm:provider_error" in result["signals"]
    assert "offline" not in json.dumps(result)


def test_conversation_followup_fallback(dataframe) -> None:
    provider = RecordingProvider({
        "supported": True,
        "modification": "REPLACE_MEASURE",
        "replace_measure_term": "salary",
        "replace_dimension_term": None,
        "aggregation": None,
        "comparison": None,
        "requested_chart": None,
        "limit": None,
        "filter": None,
        "temporal_terms": [],
        "unsupported_reason": None,
    })
    orchestrator = ConversationOrchestrator(llm_provider=provider)
    first = orchestrator.start("dataset-a", dataframe, "show sales by department")
    second = orchestrator.continue_conversation(
        "dataset-a",
        first.conversation_id,
        dataframe,
        "could you make that based on what they earn instead?",
    )
    state = orchestrator.store.get("dataset-a", first.conversation_id).active_query
    assert provider.calls == 1
    assert second.status == "READY"
    assert state.measure == "salary"
    assert state.dimension == "department"


def test_explicit_deterministic_followup_does_not_call_llm(dataframe) -> None:
    provider = RecordingProvider(None)
    orchestrator = ConversationOrchestrator(llm_provider=provider)
    first = orchestrator.start("dataset-a", dataframe, "show salary by department")
    second = orchestrator.continue_conversation(
        "dataset-a", first.conversation_id, dataframe, "use average instead"
    )
    assert second.status == "READY"
    assert provider.calls == 0


def test_repeated_mocked_interpretation_is_deterministic(dataframe) -> None:
    outputs = []
    for _ in range(2):
        provider = RecordingProvider(_query())
        outputs.append(build_visualization_results(
            dataframe,
            "give me a breakdown of what people earn across each department",
            llm_provider=provider,
        ))
    assert outputs[0] == outputs[1]


def test_strict_schema_itself_forbids_extra_fields() -> None:
    with pytest.raises(Exception):
        LLMQueryInterpretation.model_validate(_query(answer="42"))
