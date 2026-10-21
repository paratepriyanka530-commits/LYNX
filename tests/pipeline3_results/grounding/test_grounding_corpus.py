from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.pipeline3_results.grounding.dataset_knowledge import build_dataset_knowledge
from pipelines.pipeline3_results.grounding.engine import ground_semantic_intent
from pipelines.pipeline3_results.nlp.semantic_intent import SemanticIntent


CORPUS = json.loads(
    (Path(__file__).parent / "fixtures" / "grounding_corpus.json").read_text(encoding="utf-8")
)


def _report(dataset: dict) -> dict:
    columns = []
    profiles = []
    for name, dtype, semantic_type, role, description in dataset["columns"]:
        columns.append({
            "column_name": name,
            "semantic_type": semantic_type,
            "semantic_role": role,
            "value_function": (
                "entity_reference" if semantic_type == "identifier"
                else "temporal" if role == "timestamp"
                else "measurement" if role == "measurement"
                else "classification"
            ),
            "numeric_meaning": "magnitude" if role == "measurement" else None,
            "concept_description": description,
            "confidence": 0.91,
        })
        profiles.append({"column_name": name, "physical_dtype": dtype})
    return {
        "semantic_profile": {
            "dataset_context": {
                "row_entity": dataset["row_entity"],
                "row_entity_confidence": 0.9,
                "dataset_description": f"One row represents a {dataset['row_entity']}.",
            },
            "columns": columns,
        },
        "before_profile": {"column_profiles": profiles},
        "quality_summary": {"quality_score": 0.95},
    }


def _intent(payload: dict) -> SemanticIntent:
    return SemanticIntent(intent="INSIGHT", **payload)


def _tuple(expression) -> list:
    return [expression.kind, expression.operation, expression.column]


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["id"])
def test_model_independent_grounding_corpus(case: dict) -> None:
    knowledge = build_dataset_knowledge(_report(CORPUS["datasets"][case["dataset"]]))
    result = ground_semantic_intent(_intent(case["intent"]), knowledge)

    assert result.status == case["status"], result.model_dump()
    if "measure" in case:
        assert _tuple(result.measures[0].resolved_expression) == case["measure"]
    if "dimension" in case:
        assert _tuple(result.dimensions[0].resolved_expression) == case["dimension"]
    if "temporal" in case:
        expression = result.temporal[0].resolved_expression
        assert [expression.kind, expression.operation, expression.column, expression.time_granularity] == case["temporal"]
    if "candidate_columns" in case:
        candidates = result.ambiguities[0].candidates
        assert sorted({item.expression.column for item in candidates}) == case["candidate_columns"]
    if "candidate_operations" in case:
        candidates = result.ambiguities[0].candidates
        assert sorted({item.expression.operation for item in candidates}) == case["candidate_operations"]
    if "excluded_measure" in case and result.measures:
        eligible = {
            item.expression.column
            for item in result.measures[0].candidates
            if item.role_compatible and item.type_compatible
        }
        assert case["excluded_measure"] not in eligible


def test_corpus_covers_required_domains_and_twenty_cases() -> None:
    assert len(CORPUS["cases"]) >= 20
    assert {
        "retail", "hr", "healthcare", "banking", "iot",
        "cybersecurity", "manufacturing", "generic_values",
    }.issubset(CORPUS["datasets"])
