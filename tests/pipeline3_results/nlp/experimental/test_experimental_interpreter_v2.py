"""Structural-safety tests only; semantic understanding remains Qwen's job."""

from __future__ import annotations

from typing import Any, Type

import pytest
from pydantic import BaseModel, ValidationError

from intelligence.llm.provider import LLMProvider
from pipelines.pipeline3_results.nlp.experimental.evaluator_v2 import (
    LANGUAGE_CORPUS_V2,
)
from pipelines.pipeline3_results.nlp.experimental.schemas_v2 import (
    AnalyticalInterpretationV2,
)
from pipelines.pipeline3_results.nlp.experimental.semantic_interpreter_v2 import (
    PretrainedSemanticInterpreterV2,
    StructuralInterpretationError,
    validate_interpretation_structure,
)


class RecordingProvider(LLMProvider):
    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.prompts: list[str] = []

    def generate_structured(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> dict[str, Any]:
        self.prompts.append(prompt)
        return response_model.model_validate(self.response).model_dump(mode="json")


def _missing() -> dict[str, Any]:
    return {"value": None, "source": "MISSING", "evidence": None}


def _valid_output(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "intent": {"value": "INSIGHT", "source": "INFERRED", "evidence": "Which category"},
        "analytical_goal": {"value": "RANK_GROUPS", "source": "INFERRED", "evidence": "most money"},
        "measures": {"value": ["revenue"], "source": "INFERRED", "evidence": "made us the most money"},
        "dimensions": {"value": ["category"], "source": "EXPLICIT", "evidence": "category"},
        "aggregation": {"value": "SUM", "source": "INFERRED", "evidence": "made us the most money"},
        "comparison": {"value": "HIGHEST", "source": "INFERRED", "evidence": "most"},
        "limit": _missing(),
        "requested_chart": _missing(),
        "filters": _missing(),
        "time_expression": _missing(),
        "ambiguities": [],
        "missing_slots": [],
        "requires_clarification": False,
        "clarification_question": None,
        "unsupported": False,
        "unsupported_reason": None,
        "interpretation_summary": "Ranks categories by inferred revenue.",
    }
    payload.update(updates)
    return payload


def test_extra_fields_are_rejected() -> None:
    payload = _valid_output(final_answer="Category A")
    with pytest.raises(ValidationError):
        AnalyticalInterpretationV2.model_validate(payload)


def test_missing_field_requires_null_value_and_evidence() -> None:
    payload = _valid_output(limit={"value": 1, "source": "MISSING", "evidence": None})
    with pytest.raises(ValidationError, match="MISSING fields require an empty value"):
        AnalyticalInterpretationV2.model_validate(payload)


def test_missing_slot_is_structural_and_distinct_from_ambiguity() -> None:
    payload = _valid_output(
        measures=_missing(),
        analytical_goal={"value": "RANK_GROUPS", "source": "INFERRED", "evidence": "performs best"},
        aggregation={"value": "NONE", "source": "INFERRED", "evidence": "performs best"},
        comparison={"value": "HIGHEST", "source": "INFERRED", "evidence": "best"},
        missing_slots=[{"slot": "measure", "reason": "Metric absent."}],
        requires_clarification=True,
        clarification_question="Which metric should I use?",
    )
    result = AnalyticalInterpretationV2.model_validate(payload)
    validate_interpretation_structure("Which category performs best?", result)
    assert result.measures.source.value == "MISSING"
    assert result.ambiguities == []


def test_ambiguity_preserves_present_concept() -> None:
    payload = _valid_output(
        measures={"value": ["value"], "source": "EXPLICIT", "evidence": "value"},
        ambiguities=[{
            "slot": "measure",
            "concept": "value",
            "reason": "Value has multiple plausible analytical meanings.",
        }],
        requires_clarification=True,
        clarification_question="What does value mean here?",
    )
    result = AnalyticalInterpretationV2.model_validate(payload)
    validate_interpretation_structure(
        "Which category made us the most money and has the highest value?", result
    )
    assert result.measures.value == ["value"]
    assert result.missing_slots == []


def test_non_explicit_limit_and_chart_are_rejected() -> None:
    with pytest.raises(ValidationError, match="numeric limit may only be EXPLICIT"):
        AnalyticalInterpretationV2.model_validate(_valid_output(
            limit={"value": 1, "source": "INFERRED", "evidence": "most"}
        ))
    with pytest.raises(ValidationError, match="requested chart may only be EXPLICIT"):
        AnalyticalInterpretationV2.model_validate(_valid_output(
            requested_chart={"value": "BAR", "source": "INFERRED", "evidence": "show"}
        ))


def test_explicit_chart_evidence_must_name_chart() -> None:
    result = AnalyticalInterpretationV2.model_validate(_valid_output(
        requested_chart={"value": "BAR", "source": "EXPLICIT", "evidence": "show"}
    ))
    with pytest.raises(StructuralInterpretationError, match="explicitly name") as caught:
        validate_interpretation_structure(
            "Which category made us the most money? show it", result
        )
    assert caught.value.result is result


def test_explicit_limit_requires_matching_number_in_evidence() -> None:
    result = AnalyticalInterpretationV2.model_validate(_valid_output(
        limit={"value": 5, "source": "EXPLICIT", "evidence": "top products"}
    ))
    with pytest.raises(StructuralInterpretationError, match="numeric limit"):
        validate_interpretation_structure(
            "Which category made us the most money? top products", result
        )


def test_evidence_must_be_copied_from_query() -> None:
    result = AnalyticalInterpretationV2.model_validate(_valid_output())
    with pytest.raises(StructuralInterpretationError, match="exact phrase"):
        validate_interpretation_structure("Unrelated query", result)


def test_interpreter_calls_provider_once_without_dataset_context() -> None:
    provider = RecordingProvider(_valid_output())
    result = PretrainedSemanticInterpreterV2(provider).interpret(
        "Which category made us the most money?"
    )
    assert result.measures.value == ["revenue"]
    assert len(provider.prompts) == 1
    prompt = provider.prompts[0].casefold()
    assert "dataframe columns" in prompt
    assert "total_price" not in prompt
    assert "pizza_category" not in prompt


def test_schema_has_no_execution_or_answer_fields() -> None:
    schema_text = str(AnalyticalInterpretationV2.model_json_schema()).casefold()
    for forbidden in (
        "physical_column", "dataframe_column", "python", "pandas", "sql",
        "final_answer", "result_value", "chain_of_thought",
    ):
        assert forbidden not in schema_text


def test_full_language_corpus_has_forty_unique_queries() -> None:
    assert len(LANGUAGE_CORPUS_V2) == 40
    assert len(set(LANGUAGE_CORPUS_V2)) == 40
