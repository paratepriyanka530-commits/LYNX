from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pytest

from pipelines.pipeline3_results.grounding.dataset_knowledge import build_dataset_knowledge
from pipelines.pipeline3_results.grounding.engine import ground_semantic_intent
from pipelines.pipeline3_results.nlp.semantic_intent import SemanticIntent
from pipelines.pipeline3_results.planning.planner import build_analytical_plan


ROOT = Path(__file__).parents[1]
CASES = json.loads((Path(__file__).parent / "fixtures" / "analytical_plan_corpus.json").read_text(encoding="utf-8"))
DATASETS = json.loads((ROOT / "grounding" / "fixtures" / "grounding_corpus.json").read_text(encoding="utf-8"))["datasets"]


def _report(dataset: dict) -> dict:
    columns, profiles = [], []
    for name, dtype, semantic_type, role, description in dataset["columns"]:
        columns.append({
            "column_name": name,
            "semantic_type": semantic_type,
            "semantic_role": role,
            "value_function": "entity_reference" if semantic_type == "identifier" else "temporal" if role == "timestamp" else "measurement" if role == "measurement" else "location" if semantic_type == "location" else "classification",
            "numeric_meaning": "magnitude" if role == "measurement" else "not_applicable",
            "concept_description": description,
            "confidence": 0.91,
        })
        profiles.append({"column_name": name, "physical_dtype": dtype})
    return {
        "semantic_profile": {"dataset_context": {"row_entity": dataset["row_entity"], "row_entity_confidence": 0.9}, "columns": columns},
        "before_profile": {"column_profiles": profiles},
    }


def _expression_tuple(expression) -> list:
    return [expression.kind, expression.operation, expression.column]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_analytical_plan_corpus(case: dict) -> None:
    semantic = SemanticIntent(intent="INSIGHT", **case["intent"])
    knowledge = build_dataset_knowledge(_report(DATASETS[case["dataset"]]))
    grounding = ground_semantic_intent(semantic, knowledge)
    result = build_analytical_plan(semantic, grounding, knowledge)

    assert result.status == case["status"], result.model_dump()
    assert result.provenance is not None
    assert result.provenance.grounding_status == grounding.status
    if result.status != "READY":
        assert result.plan is None
        return
    plan = result.plan
    assert plan is not None
    assert plan.operation == case["operation"]
    assert "requested_chart" not in plan.provenance.source_intent
    if "measure" in case:
        assert _expression_tuple(plan.measures[0]) == case["measure"]
    if "measure_columns" in case:
        assert sorted(item.column for item in plan.measures) == case["measure_columns"]
    if "measure_count" in case:
        assert len(plan.measures) == case["measure_count"]
    if "dimension" in case:
        assert plan.dimensions[0].column == case["dimension"]
    if "temporal" in case:
        assert [plan.temporal[0].kind, plan.temporal[0].column, plan.temporal[0].time_granularity] == case["temporal"]
    if "sort" in case:
        assert plan.sort[0].direction == case["sort"]
    if "limit" in case:
        assert plan.limit == case["limit"]
    if "filter" in case:
        item = plan.filters[0]
        value = item.value.isoformat() if isinstance(item.value, date) else item.value
        assert [item.column, item.operator, item.value_type, value] == case["filter"]


def test_corpus_has_required_case_count() -> None:
    assert len(CASES) >= 25
