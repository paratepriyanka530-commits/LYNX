"""Focused tests for the isolated pretrained-first comparison prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Type

import pandas as pd
from pydantic import BaseModel

from intelligence.llm.provider import LLMProvider
from pipelines.pipeline3_results.nlp.experimental.evaluator import TEST_CORPUS
from pipelines.pipeline3_results.nlp.experimental.grounding import (
    ground_interpretation,
)
from pipelines.pipeline3_results.nlp.experimental.schemas import (
    AnalyticalInterpretation,
)
from pipelines.pipeline3_results.nlp.experimental.semantic_interpreter import (
    PretrainedSemanticInterpreter,
)


ROOT = Path(__file__).resolve().parents[4]


class RecordingProvider(LLMProvider):
    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.prompts: list[str] = []

    def generate_structured(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> dict[str, Any]:
        self.prompts.append(prompt)
        return response_model.model_validate(self.response).model_dump()


def _valid_response(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "intent": "INSIGHT",
        "analytical_goal": "RANK_GROUPS",
        "measure_concepts": ["revenue"],
        "dimension_concepts": ["category"],
        "aggregation": "SUM",
        "comparison": "HIGHEST",
        "limit": None,
        "requested_chart": None,
        "filters": [],
        "time_expression": None,
        "requires_clarification": False,
        "clarification_reason": None,
        "unsupported": False,
        "unsupported_reason": None,
        "reasoning_summary": "Ranks categories by summed revenue.",
    }
    payload.update(updates)
    return payload


def test_stage_one_prompt_contains_query_but_no_dataset_schema() -> None:
    provider = RecordingProvider(_valid_response())
    result = PretrainedSemanticInterpreter(provider).interpret(
        "Which category made us the most money?"
    )

    assert result.measure_concepts == ["revenue"]
    assert len(provider.prompts) == 1
    prompt = provider.prompts[0]
    assert "Which category made us the most money?" in prompt
    assert "total_price" not in prompt
    assert "pizza_category" not in prompt
    assert "No dataset columns or rows are available" in prompt


def test_schema_requires_reason_for_clarification() -> None:
    payload = _valid_response(
        requires_clarification=True,
        clarification_reason="The performance metric is unspecified.",
        measure_concepts=[],
    )
    result = AnalyticalInterpretation.model_validate(payload)
    assert result.requires_clarification is True


def test_schema_preserves_chart_intent_inconsistency_for_manual_evaluation() -> None:
    payload = _valid_response(requested_chart="PIE")
    result = AnalyticalInterpretation.model_validate(payload)
    assert result.intent == "INSIGHT"
    assert result.requested_chart == "PIE"


def test_grounding_reuses_existing_deterministic_evidence_path() -> None:
    dataframe = pd.read_csv(ROOT / "tests" / "datasets" / "valid" / "pizza_sales.csv")
    report = json.loads((
        ROOT / "tests" / "integration" / "pipeline1_architecture_b_pizza_output_v3"
        / "cleaning_report.json"
    ).read_text(encoding="utf-8"))
    interpretation = AnalyticalInterpretation.model_validate(_valid_response())

    grounded = ground_interpretation(
        interpretation,
        dataframe,
        pipeline1_report=report,
    )

    assert grounded.status == "RESOLVED"
    assert grounded.resolved_measures == ["total_price"]
    assert grounded.resolved_dimensions == ["pizza_category"]
    assert grounded.concepts[0].evidence


def test_grounding_preserves_model_clarification_while_grounding_known_dimension() -> None:
    dataframe = pd.DataFrame({
        "department": ["A", "B"],
        "salary": [10.0, 20.0],
        "performance_rating": [3.0, 4.0],
    })
    interpretation = AnalyticalInterpretation.model_validate(_valid_response(
        measure_concepts=[],
        dimension_concepts=["department"],
        aggregation="NONE",
        requires_clarification=True,
        clarification_reason="The performance metric is unspecified.",
    ))

    grounded = ground_interpretation(interpretation, dataframe)

    assert grounded.status == "NEEDS_CLARIFICATION"
    assert grounded.resolved_dimensions == ["department"]
    assert grounded.resolved_measures == []


def test_doughnut_is_adapted_without_changing_model_meaning() -> None:
    dataframe = pd.DataFrame({"category": ["A", "B"], "revenue": [1.0, 2.0]})
    interpretation = AnalyticalInterpretation.model_validate(_valid_response(
        intent="VISUALIZATION",
        analytical_goal="PART_TO_WHOLE",
        requested_chart="DOUGHNUT",
    ))
    grounded = ground_interpretation(interpretation, dataframe)
    assert grounded.status == "RESOLVED"
    assert interpretation.requested_chart == "DOUGHNUT"


def test_corpus_has_all_thirty_requests_and_cross_domain_coverage() -> None:
    assert len(TEST_CORPUS) == 30
    dataset_keys = {case.dataset_key for case in TEST_CORPUS}
    assert {"retail", "hr", "healthcare", "banking", "iot"} <= dataset_keys


def test_interpreter_module_does_not_contain_dataset_specific_alias_table() -> None:
    source = (
        ROOT / "pipelines" / "pipeline3_results" / "nlp" / "experimental"
        / "semantic_interpreter.py"
    ).read_text(encoding="utf-8").casefold()
    assert "pizza_category" not in source
    assert "total_price" not in source
    assert "annual_salary" not in source
