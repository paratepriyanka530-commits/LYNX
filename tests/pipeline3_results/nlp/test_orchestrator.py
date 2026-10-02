"""Tests for the canonical deterministic shared-NLP orchestrator."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipelines.pipeline3_results.nlp.orchestrator import (
    interpret_query,
    orchestrate_query,
)


def _frame(*columns: str) -> pd.DataFrame:
    return pd.DataFrame({column: [1, 2, 3] for column in columns})


def _decisions(result: object) -> dict[str, object]:
    return {item.term: item for item in result.ambiguity_decisions}  # type: ignore[attr-defined]


def test_average_salary_by_department_is_fully_resolved() -> None:
    result = interpret_query(
        "show average salary by department", _frame("salary", "department")
    )

    assert result.status == "RESOLVED"
    assert result.query.intent == "VISUALIZATION"
    assert result.intent_confidence == "MEDIUM"
    assert result.context_used is False
    assert result.query.aggregation == "MEAN"
    assert result.query.resolved_measures == ["salary"]
    assert result.query.resolved_dimensions == ["department"]


def test_highest_salary_question_is_resolved_insight() -> None:
    result = interpret_query(
        "which department has the highest salary?",
        _frame("salary", "department"),
    )

    assert result.status == "RESOLVED"
    assert result.query.intent == "INSIGHT"
    assert result.query.comparison == "HIGHEST"
    assert result.query.resolved_measures == ["salary"]
    assert result.query.resolved_dimensions == ["department"]


def test_ambiguous_sales_preserves_resolved_region() -> None:
    result = interpret_query(
        "show sales by region",
        _frame("gross_sales", "net_sales", "region"),
    )
    decisions = _decisions(result)

    assert result.status == "NEEDS_CLARIFICATION"
    assert result.query.resolved_dimensions == ["region"]
    assert result.query.resolved_measures == []
    assert decisions["sales"].choices == ["gross_sales", "net_sales"]  # type: ignore[union-attr]
    assert decisions["region"].resolved_column == "region"  # type: ignore[union-attr]
    assert result.choices == ["gross_sales", "net_sales"]


def test_invalid_salary_preserves_resolved_department() -> None:
    result = interpret_query(
        "show salary by department", _frame("age", "height", "department")
    )
    decisions = _decisions(result)

    assert result.status == "INVALID"
    assert result.query.resolved_dimensions == ["department"]
    assert decisions["salary"].status == "INVALID"  # type: ignore[union-attr]
    assert decisions["department"].status == "RESOLVED"  # type: ignore[union-attr]


def test_relationship_fields_are_independently_resolved() -> None:
    result = interpret_query(
        "relationship between discount and revenue",
        _frame("discount", "revenue"),
    )

    assert result.status == "RESOLVED"
    assert result.query.resolved_measures == ["discount", "revenue"]
    assert [item.kind for item in result.ambiguity_decisions] == [
        "RELATIONSHIP", "RELATIONSHIP"
    ]


def test_time_ambiguity_is_preserved() -> None:
    result = interpret_query(
        "sales over time", _frame("sales", "order_date", "created_at")
    )
    decisions = _decisions(result)

    assert result.status == "NEEDS_CLARIFICATION"
    assert result.query.resolved_measures == ["sales"]
    assert result.query.resolved_time_columns == []
    assert decisions["over time"].choices == ["order_date", "created_at"]  # type: ignore[union-attr]


def test_filter_field_resolves_and_value_is_preserved_without_execution() -> None:
    result = interpret_query(
        "sales in north region", _frame("sales", "customer_region")
    )

    assert result.status == "RESOLVED"
    assert result.query.resolved_measures == ["sales"]
    assert len(result.query.filters) == 1
    assert result.query.filters[0].column_term == "region"
    assert result.query.filters[0].resolved_column == "customer_region"
    assert result.query.filters[0].value_term == "north"
    assert result.query.filters[0].operator == "EQUALS"


def test_opaque_columns_resolve_through_pipeline1_semantics() -> None:
    report = {
        "semantic_profile": {
            "columns": [
                {
                    "column_name": "amt",
                    "concept_description": "Annual employee salary",
                    "semantic_type": "numeric_measurement",
                },
                {
                    "column_name": "dept_cd",
                    "concept_description": "Employee department name",
                    "semantic_type": "categorical",
                },
            ]
        }
    }

    result = interpret_query(
        "salary by department", _frame("amt", "dept_cd"), report
    )

    assert result.status == "RESOLVED"
    assert result.query.resolved_measures == ["amt"]
    assert result.query.resolved_dimensions == ["dept_cd"]


def test_missing_pipeline1_report_keeps_name_based_resolution() -> None:
    result = interpret_query(
        "show salary by department",
        _frame("salary", "department"),
        pipeline1_report=None,
    )

    assert result.status == "RESOLVED"


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"semantic_profile": {"columns": [None, "bad", 3]}},
        {"semantic_profile": {"columns": {"salary": "bad"}}},
    ],
)
def test_malformed_pipeline1_report_is_safe(report: object) -> None:
    result = interpret_query(
        "show salary by department",
        _frame("salary", "department"),
        pipeline1_report=report,
    )

    assert result.status == "RESOLVED"


def test_vague_unknown_intent_returns_safe_clarification_without_entities() -> None:
    result = interpret_query("tell me something", _frame("salary", "department"))

    assert result.status == "NEEDS_CLARIFICATION"
    assert result.query.intent == "UNKNOWN"
    assert result.query.measure_terms == []
    assert result.query.dimension_terms == []
    assert result.query.resolutions == []
    assert result.ambiguity_decisions == []
    assert result.message == "no_analytical_entities"


@pytest.mark.parametrize("text", ["", "   \t\n "])
def test_empty_or_whitespace_request_returns_controlled_result(text: str) -> None:
    result = interpret_query(text, _frame("salary"))

    assert result.status == "NEEDS_CLARIFICATION"
    assert result.query.intent == "UNKNOWN"
    assert result.query.normalized_text == ""
    assert result.message == "empty_request"


def test_zero_column_dataframe_returns_controlled_invalid_result() -> None:
    result = interpret_query("show salary by department", pd.DataFrame())

    assert result.status == "INVALID"
    assert len(result.ambiguity_decisions) == 2
    assert all(item.status == "INVALID" for item in result.ambiguity_decisions)


def test_visualization_context_routes_ambiguous_by_fragment() -> None:
    result = interpret_query(
        "sales by region", _frame("sales", "region"), context="visualization"
    )

    assert result.status == "RESOLVED"
    assert result.query.intent == "VISUALIZATION"
    assert result.intent_confidence == "LOW"
    assert result.context_used is True
    assert "intent:context_fallback" in result.signals


def test_explicit_insight_wording_overrides_visualization_context() -> None:
    result = interpret_query(
        "which region has the highest sales?",
        _frame("sales", "region"),
        context="visualization",
    )

    assert result.query.intent == "INSIGHT"
    assert result.status == "RESOLVED"


def test_explicit_visualization_wording_overrides_insights_context() -> None:
    result = interpret_query(
        "plot sales by region",
        _frame("sales", "region"),
        context="insights",
    )

    assert result.query.intent == "VISUALIZATION"
    assert result.status == "RESOLVED"


def test_audit_signals_preserve_each_stage_origin() -> None:
    result = interpret_query(
        "show average salary by department", _frame("salary", "department")
    )

    assert "normalize:text_normalized" in result.signals
    assert "intent:show_by_pattern" in result.signals
    assert "extract:explicit_aggregation" in result.signals
    assert "extract:by_pattern" in result.signals
    assert "resolver:exact_normalized_name" in result.signals
    assert "ambiguity:clear_exact_match" in result.signals
    assert "ambiguity:overall_resolved" in result.signals
    assert "orchestrator:resolved" in result.signals


def test_repeated_orchestration_is_deterministic() -> None:
    frame = _frame("gross_sales", "net_sales", "region")
    expected = interpret_query("show sales by region", frame)

    for _ in range(10):
        assert interpret_query("show sales by region", frame) == expected
        assert orchestrate_query("show sales by region", frame) == expected


def test_inputs_are_not_mutated() -> None:
    frame = _frame("amt", "dept_cd")
    frame_before = frame.copy(deep=True)
    report = {
        "semantic_profile": {"columns": [
            {"column_name": "amt", "concept_description": "Employee salary"},
            {"column_name": "dept_cd", "concept_description": "Department name"},
        ]}
    }
    report_before = json.loads(json.dumps(report))

    interpret_query("salary by department", frame, report)

    pd.testing.assert_frame_equal(frame, frame_before)
    assert report == report_before


def test_query_interpretation_is_json_serializable() -> None:
    result = interpret_query(
        "show sales by region",
        _frame("gross_sales", "net_sales", "region"),
    )

    payload = result.model_dump(mode="json")
    encoded = json.dumps(payload)

    assert json.loads(encoded) == payload
    assert '"ambiguity_decisions"' in encoded


def test_no_fallback_hallucination_for_unsupported_concepts() -> None:
    result = interpret_query("profit by city", _frame("revenue", "region"))

    assert result.status == "INVALID"
    assert result.query.resolved_measures == []
    assert result.query.resolved_dimensions == []
    assert all(item.resolved_column is None for item in result.ambiguity_decisions)
