"""Tests for deterministic Phase 5 ambiguity detection."""

from __future__ import annotations

import json

import pytest

from pipelines.pipeline3_results.nlp.ambiguity_detector import (
    detect_ambiguities,
    detect_ambiguity,
)
from pipelines.pipeline3_results.nlp.schemas import ColumnCandidate, TermResolution


def _candidate(
    name: str,
    score: float,
    *sources: str,
) -> ColumnCandidate:
    return ColumnCandidate(
        column_name=name,
        score=score,
        evidence_sources=list(sources),
    )


def _resolution(
    term: str,
    *candidates: ColumnCandidate,
    kind: str = "MEASURE",
) -> TermResolution:
    return TermResolution(
        term=term,
        kind=kind,  # type: ignore[arg-type]
        candidates=list(candidates),
    )


def test_single_exact_name_candidate_is_resolved() -> None:
    decision = detect_ambiguity(
        _resolution("salary", _candidate("salary", 1.0, "exact_normalized_name"))
    )

    assert decision.status == "RESOLVED"
    assert decision.resolved_column == "salary"
    assert decision.choices == []
    assert decision.reason == "clear_exact_match"
    assert "single_supported_candidate" in decision.signals
    assert "clear_exact_match" in decision.signals


def test_exact_name_winner_beats_weaker_alias() -> None:
    decision = detect_ambiguity(_resolution(
        "salary",
        _candidate("salary", 1.0, "exact_normalized_name"),
        _candidate("annual_salary_backup", 0.66, "alias_match:salary"),
    ))

    assert decision.status == "RESOLVED"
    assert decision.resolved_column == "salary"
    assert decision.reason == "clear_exact_match"
    assert "clear_exact_match" in decision.signals


def test_exact_token_winner_beats_weaker_semantic_match() -> None:
    decision = detect_ambiguity(_resolution(
        "salary",
        _candidate("annual_salary", 0.86, "exact_token_match"),
        _candidate("amt", 0.76, "semantic_description_match"),
    ))

    assert decision.status == "RESOLVED"
    assert decision.resolved_column == "annual_salary"
    assert decision.reason == "clear_evidence_tier_separation"
    assert "clear_tier_separation" in decision.signals


def test_equal_sales_candidates_need_clarification() -> None:
    decision = detect_ambiguity(_resolution(
        "sales",
        _candidate("gross_sales", 0.86, "exact_token_match"),
        _candidate("net_sales", 0.86, "exact_token_match"),
    ))

    assert decision.status == "NEEDS_CLARIFICATION"
    assert decision.resolved_column is None
    assert decision.choices == ["gross_sales", "net_sales"]
    assert "equal_top_candidates" in decision.signals


def test_equal_temporal_candidates_need_clarification() -> None:
    decision = detect_ambiguity(_resolution(
        "time",
        _candidate("order_date", 0.76, "semantic_description_match"),
        _candidate("created_at", 0.76, "semantic_description_match"),
        kind="TIME",
    ))

    assert decision.status == "NEEDS_CLARIFICATION"
    assert decision.choices == ["order_date", "created_at"]


def test_no_supported_salary_candidate_is_invalid() -> None:
    decision = detect_ambiguity(_resolution("salary"))

    assert decision.status == "INVALID"
    assert decision.reason == "no_supported_candidate"
    assert decision.choices == []
    assert "no_supported_candidate" in decision.signals


def test_one_weak_but_supported_candidate_is_resolved() -> None:
    decision = detect_ambiguity(_resolution(
        "sales", _candidate("total_price", 0.66, "alias_match:sales")
    ))

    assert decision.status == "RESOLVED"
    assert decision.resolved_column == "total_price"
    assert "single_supported_candidate" in decision.signals


def test_role_only_candidate_is_not_promoted() -> None:
    decision = detect_ambiguity(_resolution(
        "salary",
        _candidate("age", 0.02, "analytical_role_support:MEASURE"),
    ))

    assert decision.status == "INVALID"
    assert decision.resolved_column is None


def test_sales_ambiguous_and_region_resolved_is_overall_clarification() -> None:
    sales = _resolution(
        "sales",
        _candidate("gross_sales", 0.86, "exact_token_match"),
        _candidate("net_sales", 0.86, "exact_token_match"),
    )
    region = _resolution(
        "region",
        _candidate("customer_region", 0.86, "exact_token_match"),
        kind="DIMENSION",
    )

    result = detect_ambiguities([sales, region])

    assert result.status == "NEEDS_CLARIFICATION"
    assert [decision.status for decision in result.decisions] == [
        "NEEDS_CLARIFICATION", "RESOLVED"
    ]
    assert "multiple_required_terms" in result.signals
    assert "overall_needs_clarification" in result.signals


def test_invalid_salary_takes_precedence_over_resolved_department() -> None:
    salary = _resolution("salary")
    department = _resolution(
        "department",
        _candidate("department", 1.0, "exact_normalized_name"),
        kind="DIMENSION",
    )

    result = detect_ambiguities([salary, department])

    assert result.status == "INVALID"
    assert "overall_invalid" in result.signals


def test_all_required_terms_resolved_is_overall_resolved() -> None:
    salary = _resolution(
        "salary", _candidate("salary", 1.0, "exact_normalized_name")
    )
    department = _resolution(
        "department",
        _candidate("department", 1.0, "exact_normalized_name"),
        kind="DIMENSION",
    )

    result = detect_ambiguities([salary, department])

    assert result.status == "RESOLVED"
    assert result.reason == "all_required_terms_resolved"
    assert "overall_resolved" in result.signals


def test_both_relationship_fields_resolved_without_orientation() -> None:
    discount = _resolution(
        "discount",
        _candidate("discount", 1.0, "exact_normalized_name"),
        kind="RELATIONSHIP",
    )
    revenue = _resolution(
        "revenue",
        _candidate("revenue", 1.0, "exact_normalized_name"),
        kind="RELATIONSHIP",
    )

    result = detect_ambiguities([discount, revenue])

    assert result.status == "RESOLVED"
    assert [item.resolved_column for item in result.decisions] == [
        "discount", "revenue"
    ]


def test_one_ambiguous_relationship_field_requires_clarification() -> None:
    discount = _resolution(
        "discount",
        _candidate("discount", 1.0, "exact_normalized_name"),
        kind="RELATIONSHIP",
    )
    revenue = _resolution(
        "revenue",
        _candidate("gross_revenue", 0.86, "exact_token_match"),
        _candidate("net_revenue", 0.86, "exact_token_match"),
        kind="RELATIONSHIP",
    )

    result = detect_ambiguities([discount, revenue])

    assert result.status == "NEEDS_CLARIFICATION"
    assert result.decisions[1].choices == ["gross_revenue", "net_revenue"]


def test_filter_field_ambiguity_requires_clarification() -> None:
    decision = detect_ambiguity(_resolution(
        "region",
        _candidate("customer_region", 0.86, "exact_token_match"),
        _candidate("shipping_region", 0.86, "exact_token_match"),
        kind="FILTER",
    ))

    assert decision.status == "NEEDS_CLARIFICATION"
    assert decision.choices == ["customer_region", "shipping_region"]


def test_clarification_excludes_irrelevant_weaker_candidates() -> None:
    decision = detect_ambiguity(_resolution(
        "sales",
        _candidate("net_sales", 0.86, "exact_token_match"),
        _candidate("gross_sales", 0.86, "exact_token_match"),
        _candidate("total_price", 0.66, "alias_match:sales"),
    ))

    assert decision.choices == ["net_sales", "gross_sales"]


def test_column_order_does_not_break_semantic_tie() -> None:
    first = detect_ambiguity(_resolution(
        "sales",
        _candidate("gross_sales", 0.86, "exact_token_match"),
        _candidate("net_sales", 0.86, "exact_token_match"),
    ))
    reversed_input = detect_ambiguity(_resolution(
        "sales",
        _candidate("net_sales", 0.86, "exact_token_match"),
        _candidate("gross_sales", 0.86, "exact_token_match"),
    ))

    assert first.status == reversed_input.status == "NEEDS_CLARIFICATION"
    assert first.resolved_column is reversed_input.resolved_column is None
    assert first.choices == ["gross_sales", "net_sales"]
    assert reversed_input.choices == ["net_sales", "gross_sales"]


def test_same_tier_role_bonus_difference_is_still_near_equal() -> None:
    decision = detect_ambiguity(_resolution(
        "sales",
        _candidate(
            "gross_sales", 0.88,
            "exact_token_match", "analytical_role_support:MEASURE",
        ),
        _candidate("net_sales", 0.86, "exact_token_match"),
    ))

    assert decision.status == "NEEDS_CLARIFICATION"
    assert decision.choices == ["gross_sales", "net_sales"]
    assert "near_equal_candidates" in decision.signals


def test_repeated_detection_is_deterministic() -> None:
    resolution = _resolution(
        "sales",
        _candidate("gross_sales", 0.86, "exact_token_match"),
        _candidate("net_sales", 0.86, "exact_token_match"),
    )

    outputs = [detect_ambiguity(resolution).model_dump() for _ in range(5)]

    assert all(output == outputs[0] for output in outputs)


def test_decision_and_overall_result_are_json_serializable() -> None:
    resolution = _resolution(
        "salary", _candidate("salary", 1.0, "exact_normalized_name")
    )
    decision = detect_ambiguity(resolution)
    overall = detect_ambiguities([resolution])

    decision_json = json.dumps(decision.model_dump(mode="json"))
    overall_json = json.dumps(overall.model_dump(mode="json"))

    assert '"status": "RESOLVED"' in decision_json
    assert '"status": "RESOLVED"' in overall_json


def test_absent_optional_term_does_not_invalidate_overall_query() -> None:
    optional_time = _resolution("time", kind="TIME")
    salary = _resolution(
        "salary", _candidate("salary", 1.0, "exact_normalized_name")
    )

    result = detect_ambiguities([salary, optional_time], required=[True, False])

    assert result.status == "RESOLVED"
    assert result.decisions[1].reason == "optional_term_unsupported"
    assert result.decisions[1].required is False


def test_required_flag_count_must_match_resolution_count() -> None:
    with pytest.raises(ValueError, match="one flag per resolution"):
        detect_ambiguities([_resolution("salary")], required=[])
