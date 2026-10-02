"""Tests for deterministic, evidence-aware NLP column resolution."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipelines.pipeline3_results.nlp.column_resolver import resolve_columns
from pipelines.pipeline3_results.nlp.schemas import (
    ExtractedEntities,
    QueryFilter,
    StructuredQuery,
)
from pipelines.pipeline3_results.visualization.schemas import (
    ColumnClassification,
    DatasetClassification,
)


def _frame(*columns: str) -> pd.DataFrame:
    return pd.DataFrame({column: [1, 2, 3] for column in columns})


def _entities(
    *,
    measures: list[str] | None = None,
    dimensions: list[str] | None = None,
    times: list[str] | None = None,
    filters: list[QueryFilter] | None = None,
    signals: tuple[str, ...] = (),
) -> ExtractedEntities:
    return ExtractedEntities(
        measure_terms=measures or [],
        dimension_terms=dimensions or [],
        time_terms=times or [],
        filters=filters or [],
        signals=signals,
    )


def _pipeline1(*entries: dict[str, object]) -> dict[str, object]:
    return {"semantic_profile": {"columns": list(entries)}}


def _classification(**roles: str) -> DatasetClassification:
    columns = [
        ColumnClassification(
            name=name,
            role=role,  # type: ignore[arg-type]
            dtype="object",
            unique_count=3,
            unique_ratio=1.0,
            missing_count=0,
            missing_ratio=0.0,
        )
        for name, role in roles.items()
    ]
    return DatasetClassification(
        dataset_name="test",
        row_count=3,
        column_count=len(columns),
        columns=columns,
    )


def _names(result: list[object], index: int = 0) -> list[str]:
    return [candidate.column_name for candidate in result[index].candidates]  # type: ignore[attr-defined]


def test_exact_normalized_full_name_is_strong_match() -> None:
    result = resolve_columns(
        _entities(measures=["Employee Salary"]),
        _frame("employee_salary", "department"),
    )

    candidate = result[0].candidates[0]
    assert candidate.column_name == "employee_salary"
    assert candidate.score == 1.0
    assert "exact_normalized_name" in candidate.evidence_sources
    assert result[0].resolved_column is None


def test_subtoken_match_surfaces_compound_column() -> None:
    result = resolve_columns(
        _entities(dimensions=["region"]),
        _frame("customer_region", "total_price"),
    )

    assert _names(result) == ["customer_region"]
    assert "exact_token_match" in result[0].candidates[0].evidence_sources


def test_exact_match_ranks_above_alias_match() -> None:
    result = resolve_columns(
        _entities(measures=["sales"]),
        _frame("total_price", "sales"),
    )

    assert _names(result) == ["sales", "total_price"]
    assert result[0].candidates[0].score > result[0].candidates[1].score


def test_ambiguous_sales_candidates_are_both_retained() -> None:
    result = resolve_columns(
        _entities(measures=["sales"]),
        _frame("gross_sales", "net_sales", "region"),
    )

    assert _names(result) == ["gross_sales", "net_sales"]
    assert result[0].resolved_column is None
    assert result[0].confidence is None


def test_pipeline1_semantics_resolve_opaque_measure_name() -> None:
    report = _pipeline1({
        "column_name": "amt",
        "semantic_type": "numeric_measurement",
        "concept_description": "Annual employee salary",
        "numeric_meaning": "employee compensation",
    })

    result = resolve_columns(
        _entities(measures=["salary"]), _frame("amt", "dept"), report
    )

    assert _names(result) == ["amt"]
    candidate = result[0].candidates[0]
    assert "semantic_description_match" in candidate.evidence_sources
    assert candidate.semantic_evidence["concept_description"] == "Annual employee salary"


def test_pipeline1_semantics_resolve_opaque_dimension_name() -> None:
    report = _pipeline1({
        "column_name": "dept_cd",
        "semantic_type": "categorical",
        "concept_description": "Employee department name",
    })

    result = resolve_columns(
        _entities(dimensions=["department"]), _frame("dept_cd", "amt"), report
    )

    assert _names(result) == ["dept_cd"]
    assert "semantic_description_match" in result[0].candidates[0].evidence_sources


def test_analytical_role_alone_cannot_manufacture_salary_match() -> None:
    frame = _frame("age", "height", "weight")
    roles = _classification(age="MEASURE", height="MEASURE", weight="MEASURE")

    result = resolve_columns(
        _entities(measures=["salary"]), frame, classification=roles
    )

    assert _names(result) == []


def test_temporal_ambiguity_retains_multiple_columns() -> None:
    frame = _frame("order_date", "created_at", "sales")
    roles = _classification(
        order_date="TEMPORAL", created_at="TEMPORAL", sales="MEASURE"
    )

    result = resolve_columns(
        _entities(times=["time"]), frame, classification=roles
    )

    assert _names(result) == ["order_date", "created_at"]
    assert all(
        "analytical_role_support:TEMPORAL" in candidate.evidence_sources
        for candidate in result[0].candidates
    )


def test_identifier_role_does_not_receive_measure_support() -> None:
    result = resolve_columns(
        _entities(measures=["salary"]),
        _frame("employee_id"),
        classification=_classification(employee_id="IDENTIFIER"),
    )

    assert _names(result) == []


def test_filter_field_is_resolved_but_filter_value_is_not_validated() -> None:
    query_filter = QueryFilter(
        raw_text="region equals north",
        column_term="region",
        operator="EQUALS",
        value_term="north",
    )

    result = resolve_columns(
        _entities(filters=[query_filter]), _frame("customer_region", "sales")
    )

    assert result[0].kind == "FILTER"
    assert result[0].term == "region"
    assert _names(result) == ["customer_region"]


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"semantic_profile": {}},
        {"semantic_profile": {"columns": [None, "bad", 7]}},
        {"semantic_profile": {"columns": {"salary": "bad"}}},
    ],
)
def test_missing_or_malformed_semantic_evidence_is_safe(report: object) -> None:
    result = resolve_columns(
        _entities(measures=["salary"]), _frame("salary", "department"), report
    )

    assert _names(result) == ["salary"]


def test_missing_pipeline1_report_still_allows_name_resolution() -> None:
    result = resolve_columns(
        _entities(dimensions=["department"]), _frame("salary", "department"), None
    )

    assert _names(result) == ["department"]


def test_semantics_for_nonexistent_column_are_ignored() -> None:
    report = _pipeline1({
        "column_name": "not_in_dataframe",
        "concept_description": "Annual employee salary",
    })

    result = resolve_columns(
        _entities(measures=["salary"]), _frame("amt", "department"), report
    )

    assert _names(result) == []


def test_candidate_order_is_deterministic_across_runs() -> None:
    query = _entities(measures=["sales"])
    frame = _frame("gross_sales", "net_sales", "sales_amount")

    runs = [_names(resolve_columns(query, frame)) for _ in range(5)]

    assert all(run == runs[0] for run in runs)


def test_resolver_does_not_mutate_input_dataframe_or_report() -> None:
    frame = _frame("amt", "dept")
    frame_before = frame.copy(deep=True)
    report = _pipeline1({
        "column_name": "amt", "concept_description": "Annual employee salary"
    })
    report_before = json.loads(json.dumps(report))

    resolve_columns(_entities(measures=["salary"]), frame, report)

    pd.testing.assert_frame_equal(frame, frame_before)
    assert report == report_before


def test_original_dataframe_order_breaks_equal_strength_ties() -> None:
    query = _entities(measures=["sales"])

    forward = resolve_columns(query, _frame("net_sales", "gross_sales"))
    reverse = resolve_columns(query, _frame("gross_sales", "net_sales"))

    assert _names(forward) == ["net_sales", "gross_sales"]
    assert _names(reverse) == ["gross_sales", "net_sales"]


def test_candidate_evidence_is_json_serializable() -> None:
    report = _pipeline1({
        "column_name": "amt",
        "concept_description": "Annual employee salary",
        "detected_patterns": ["currency", "positive values"],
    })
    result = resolve_columns(
        _entities(measures=["salary"]), _frame("amt"), report
    )

    encoded = json.dumps([item.model_dump(mode="json") for item in result])

    assert '"column_name": "amt"' in encoded


def test_structured_query_and_mapping_shaped_semantic_columns_are_supported() -> None:
    query = StructuredQuery(
        raw_text="salary by department",
        measure_terms=["salary"],
        dimension_terms=["department"],
    )
    report = {
        "semantic_profile": {
            "columns": {
                "amt": {"concept_description": "Annual employee salary"},
                "dept_cd": {"concept_description": "Employee department"},
            }
        }
    }

    result = resolve_columns(query, _frame("amt", "dept_cd"), report)

    assert [(item.term, _names(result, index)) for index, item in enumerate(result)] == [
        ("salary", ["amt"]),
        ("department", ["dept_cd"]),
    ]


def test_relationship_signal_marks_measure_terms_as_relationship_fields() -> None:
    result = resolve_columns(
        _entities(
            measures=["sales", "discount"],
            signals=("two_field_pattern",),
        ),
        _frame("sales", "discount"),
    )

    assert [item.kind for item in result] == ["RELATIONSHIP", "RELATIONSHIP"]


def test_generic_short_token_does_not_create_weak_matches() -> None:
    result = resolve_columns(
        _entities(dimensions=["id"]),
        _frame("employee_id", "order_id", "region"),
    )

    assert _names(result) == []


def test_unsafe_character_substring_does_not_match() -> None:
    result = resolve_columns(
        _entities(measures=["age"]), _frame("package_type", "region")
    )

    assert _names(result) == []


def test_plural_term_matches_singular_pipeline1_semantic_word() -> None:
    report = _pipeline1({
        "column_name": "order_id",
        "semantic_type": "identifier",
        "concept_description": "Unique order identifier",
    })

    result = resolve_columns(
        _entities(measures=["orders"]), _frame("order_id"), report
    )

    assert _names(result) == ["order_id"]
    assert "semantic_description_match" in result[0].candidates[0].evidence_sources
