"""Unit tests for Pipeline 3 Stage 10 -- Unified Visualization Response Contract.

Tests the response builder layer that unifies Stage 6/8 payloads with
Stage 9 deterministic factual explanations into a stable, frontend-safe contract.

Covers:
1. automatic Stage 6 payload + Stage 9 explanation
2. requested Stage 8 payload + Stage 9 explanation
3. score preservation (float preserved)
4. score=None preservation for requested visualization
5. chart_data preservation
6. explanation attachment
7. READY response
8. NO_MEANINGFUL_VISUALIZATION response
9. INVALID response + error field preservation
10. NEEDS_CLARIFICATION response + clarification_options preservation
11. missing explanation / missing chart_data handling
12. missing / empty visualization list handling
13. JSON round-trip
14. deterministic repeated execution
15. primitive JSON-safe output
16. no mutation of input payload
17. forbidden causal/speculative terminology check
18. invalid mode raises ValueError
19. non-mapping payload raises TypeError
20. visualization ordering preserved
21. SUPPORTED_MODES constant
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

import pytest

from pipelines.pipeline3_results.visualization.explanation import FORBIDDEN_WORDS
from pipelines.pipeline3_results.visualization.response_builder import (
    SUPPORTED_MODES,
    build_visualization_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_count_bar_viz(score: float | None = 0.999) -> dict[str, Any]:
    return {
        "rank": 1,
        "chart_type": "COUNT_BAR",
        "x_column": "pizza_category",
        "y_column": None,
        "score": score,
        "chart_data": {
            "chart_type": "COUNT_BAR",
            "x_column": "pizza_category",
            "y_column": None,
            "aggregation": "count",
            "points": [
                {"x": "Classic", "y": 14888},
                {"x": "Supreme", "y": 11987},
                {"x": "Veggie", "y": 11449},
                {"x": "Chicken", "y": 11050},
            ],
            "bins": [],
            "signals": ["4 category groups produced"],
            "usable_row_count": 48620,
            "sampled": False,
            "has_data": True,
        },
        "signals": ["COUNT_BAR chart resolved"],
    }


def _make_bar_viz(score: float | None = 0.95) -> dict[str, Any]:
    return {
        "rank": 2,
        "chart_type": "BAR",
        "x_column": "pizza_category",
        "y_column": "total_price",
        "score": score,
        "chart_data": {
            "chart_type": "BAR",
            "x_column": "pizza_category",
            "y_column": "total_price",
            "aggregation": "sum",
            "points": [
                {"x": "Classic", "y": 220053.1},
                {"x": "Supreme", "y": 208197.0},
                {"x": "Veggie", "y": 193690.45},
                {"x": "Chicken", "y": 195919.5},
            ],
            "bins": [],
            "signals": ["aggregated using sum via conservative default"],
            "usable_row_count": 48620,
            "sampled": False,
            "has_data": True,
        },
        "signals": ["BAR chart resolved"],
    }


def _make_requested_median_bar_viz() -> dict[str, Any]:
    return {
        "rank": 1,
        "chart_type": "BAR",
        "x_column": "pizza_category",
        "y_column": "unit_price",
        "score": None,
        "chart_data": {
            "chart_type": "BAR",
            "x_column": "pizza_category",
            "y_column": "unit_price",
            "aggregation": "median",
            "points": [
                {"x": "Classic", "y": 16.0},
                {"x": "Supreme", "y": 17.5},
                {"x": "Veggie", "y": 16.5},
                {"x": "Chicken", "y": 17.75},
            ],
            "bins": [],
            "signals": [
                "48620 valid observations used",
                "explicit MEDIAN aggregation applied to unit_price",
            ],
            "usable_row_count": 48620,
            "sampled": False,
            "has_data": True,
        },
        "signals": ["explicit MEDIAN aggregation applied to unit_price"],
    }


@pytest.fixture
def automatic_stage6_payload() -> dict[str, Any]:
    return {
        "status": "READY",
        "dataset_name": "pizza_sales",
        "visualization_count": 2,
        "visualizations": [
            _make_count_bar_viz(score=0.999),
            _make_bar_viz(score=0.95),
        ],
        "signals": ["Dataset processed successfully."],
    }


@pytest.fixture
def requested_stage8_payload() -> dict[str, Any]:
    return {
        "status": "READY",
        "dataset_name": "pizza_sales",
        "visualization_count": 1,
        "visualizations": [_make_requested_median_bar_viz()],
        "signals": ["Requested visualization resolved."],
    }


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_1_automatic_stage6_payload_with_explanation(
    automatic_stage6_payload: dict[str, Any],
) -> None:
    """1. Automatic Stage 6 payload gets Stage 9 explanation attached."""
    resp = build_visualization_response(automatic_stage6_payload, mode="automatic")
    assert resp["mode"] == "automatic"
    assert resp["status"] == "READY"
    assert resp["visualization_count"] == 2
    assert len(resp["visualizations"]) == 2

    for viz in resp["visualizations"]:
        assert "explanation" in viz
        expl = viz["explanation"]
        assert "why_this_chart" in expl
        assert "what_it_shows" in expl
        assert "aggregation" in expl
        assert "data_coverage" in expl
        assert "evidence" in expl
        assert isinstance(expl["evidence"], list)


def test_2_requested_stage8_payload_with_explanation(
    requested_stage8_payload: dict[str, Any],
) -> None:
    """2. Requested Stage 8 payload gets Stage 9 explanation with attribution."""
    resp = build_visualization_response(requested_stage8_payload, mode="requested")
    assert resp["mode"] == "requested"
    assert resp["status"] == "READY"
    assert len(resp["visualizations"]) == 1

    viz = resp["visualizations"][0]
    assert viz["score"] is None
    expl = viz["explanation"]
    assert "user-requested" in expl["aggregation"].lower()
    assert "median" in expl["aggregation"].lower()


def test_3_score_preservation_automatic(
    automatic_stage6_payload: dict[str, Any],
) -> None:
    """3. Automatic Stage 5/6 score is preserved exactly as float."""
    resp = build_visualization_response(automatic_stage6_payload, mode="automatic")
    assert resp["visualizations"][0]["score"] == pytest.approx(0.999)
    assert resp["visualizations"][1]["score"] == pytest.approx(0.95)


def test_4_score_none_preservation_requested(
    requested_stage8_payload: dict[str, Any],
) -> None:
    """4. Requested Stage 8 score=None is preserved, never invented."""
    resp = build_visualization_response(requested_stage8_payload, mode="requested")
    assert resp["visualizations"][0]["score"] is None


def test_5_chart_data_preservation(
    automatic_stage6_payload: dict[str, Any],
) -> None:
    """5. All chart_data fields and points are preserved intact."""
    resp = build_visualization_response(automatic_stage6_payload, mode="automatic")
    cd = resp["visualizations"][0]["chart_data"]
    assert cd["chart_type"] == "COUNT_BAR"
    assert cd["x_column"] == "pizza_category"
    assert cd["y_column"] is None
    assert cd["aggregation"] == "count"
    assert len(cd["points"]) == 4
    assert cd["points"][0] == {"x": "Classic", "y": 14888}
    assert cd["has_data"] is True


def test_6_explanation_attachment(
    automatic_stage6_payload: dict[str, Any],
) -> None:
    """6. Explanation sub-dict contains all 5 required conceptual keys."""
    resp = build_visualization_response(automatic_stage6_payload, mode="automatic")
    for viz in resp["visualizations"]:
        expl = viz["explanation"]
        assert isinstance(expl["why_this_chart"], str)
        assert isinstance(expl["what_it_shows"], str)
        assert isinstance(expl["aggregation"], str)
        assert isinstance(expl["data_coverage"], str)
        assert isinstance(expl["evidence"], list)
        assert len(expl["evidence"]) > 0


def test_7_ready_response(automatic_stage6_payload: dict[str, Any]) -> None:
    """7. READY response preserves status, dataset_name, count, signals."""
    resp = build_visualization_response(automatic_stage6_payload, mode="automatic")
    assert resp["status"] == "READY"
    assert resp["dataset_name"] == "pizza_sales"
    assert resp["visualization_count"] == 2
    assert "Dataset processed successfully." in resp["signals"]


def test_8_no_meaningful_visualization() -> None:
    """8. NO_MEANINGFUL_VISUALIZATION is preserved properly."""
    payload = {
        "status": "NO_MEANINGFUL_VISUALIZATION",
        "dataset_name": "empty_table",
        "visualization_count": 0,
        "visualizations": [],
        "signals": ["No valid columns found."],
    }
    resp = build_visualization_response(payload, mode="automatic")
    assert resp["status"] == "NO_MEANINGFUL_VISUALIZATION"
    assert resp["visualization_count"] == 0
    assert resp["visualizations"] == []
    assert resp["dataset_name"] == "empty_table"


def test_9_invalid_response_preservation() -> None:
    """9. INVALID payload preserves status and errors field."""
    payload = {
        "status": "INVALID",
        "dataset_name": "test_ds",
        "visualization_count": 0,
        "visualizations": [],
        "signals": ["Validation failed."],
        "errors": ["Column 'missing_col' does not exist."],
    }
    resp = build_visualization_response(payload, mode="requested")
    assert resp["status"] == "INVALID"
    assert resp["mode"] == "requested"
    assert "errors" in resp
    assert "Column 'missing_col' does not exist." in resp["errors"]


def test_10_needs_clarification_response_preservation() -> None:
    """10. NEEDS_CLARIFICATION payload preserves status and clarification_options."""
    payload = {
        "status": "NEEDS_CLARIFICATION",
        "dataset_name": "test_ds",
        "visualization_count": 0,
        "visualizations": [],
        "signals": ["Ambiguous request."],
        "clarification_options": ["Did you mean 'unit_price' or 'total_price'?"],
    }
    resp = build_visualization_response(payload, mode="requested")
    assert resp["status"] == "NEEDS_CLARIFICATION"
    assert resp["mode"] == "requested"
    assert "clarification_options" in resp
    assert len(resp["clarification_options"]) == 1


def test_11_missing_explanation_and_chart_data_handling() -> None:
    """11. Missing chart_data gracefully falls back to default explanation without crashing."""
    payload = {
        "status": "READY",
        "dataset_name": "edge_ds",
        "visualization_count": 1,
        "visualizations": [
            {
                "rank": 1,
                "chart_type": "BAR",
                "x_column": "cat",
                "y_column": "val",
                "score": None,
                "chart_data": None,
                "signals": [],
            }
        ],
        "signals": [],
    }
    resp = build_visualization_response(payload, mode="requested")
    assert resp["status"] == "READY"
    viz = resp["visualizations"][0]
    assert "explanation" in viz
    assert viz["explanation"]["why_this_chart"] == "Visualization rationale is unavailable."
    assert viz["explanation"]["what_it_shows"] == "No displayable chart data available."


def test_12_missing_visualization_handling() -> None:
    """12. Empty visualizations list is handled cleanly."""
    payload = {
        "status": "READY",
        "dataset_name": "none",
        "visualization_count": 0,
        "visualizations": [],
        "signals": [],
    }
    resp = build_visualization_response(payload, mode="automatic")
    assert resp["visualizations"] == []
    assert resp["visualization_count"] == 0


def test_13_json_round_trip(automatic_stage6_payload: dict[str, Any]) -> None:
    """13. Full response round-trips via standard json.dumps and json.loads."""
    resp = build_visualization_response(automatic_stage6_payload, mode="automatic")
    dumped = json.dumps(resp)
    loaded = json.loads(dumped)
    assert loaded["mode"] == "automatic"
    assert loaded["status"] == "READY"
    assert len(loaded["visualizations"]) == 2
    assert loaded["visualizations"][0]["rank"] == 1


def test_14_deterministic_repeated_execution(
    automatic_stage6_payload: dict[str, Any],
) -> None:
    """14. Repeated executions yield identical output."""
    resp1 = build_visualization_response(automatic_stage6_payload, mode="automatic")
    resp2 = build_visualization_response(automatic_stage6_payload, mode="automatic")
    assert resp1 == resp2


def test_15_primitive_json_safe_output(
    automatic_stage6_payload: dict[str, Any],
) -> None:
    """15. Output contains only standard Python primitives."""
    resp = build_visualization_response(automatic_stage6_payload, mode="automatic")

    def _assert_primitive(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert isinstance(k, str)
                _assert_primitive(v)
        elif isinstance(obj, list):
            for item in obj:
                _assert_primitive(item)
        else:
            assert isinstance(obj, (int, float, str, bool, type(None)))

    _assert_primitive(resp)


def test_16_no_mutation_of_input_payload(
    automatic_stage6_payload: dict[str, Any],
) -> None:
    """16. Input payload is not mutated in-place."""
    payload_copy = copy.deepcopy(automatic_stage6_payload)
    build_visualization_response(automatic_stage6_payload, mode="automatic")
    assert automatic_stage6_payload == payload_copy


def test_17_forbidden_causal_speculative_terminology(
    automatic_stage6_payload: dict[str, Any],
    requested_stage8_payload: dict[str, Any],
) -> None:
    """17. No forbidden causal or speculative terms appear in explanations."""
    for payload, mode in [
        (automatic_stage6_payload, "automatic"),
        (requested_stage8_payload, "requested"),
    ]:
        resp = build_visualization_response(payload, mode=mode)
        for viz in resp["visualizations"]:
            expl = viz["explanation"]
            for field in ("why_this_chart", "what_it_shows", "aggregation", "data_coverage"):
                text = expl.get(field, "")
                words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
                forbidden_found = set(words) & FORBIDDEN_WORDS
                assert not forbidden_found, f"Found forbidden terms {forbidden_found} in {field}: {text}"
            for ev in expl.get("evidence", []):
                words = re.findall(r"\b[a-zA-Z]+\b", ev.lower())
                forbidden_found = set(words) & FORBIDDEN_WORDS
                assert not forbidden_found, f"Found forbidden terms {forbidden_found} in evidence: {ev}"


def test_18_invalid_mode_raises_value_error(
    automatic_stage6_payload: dict[str, Any],
) -> None:
    """18. Non-supported mode string or non-string mode raises ValueError/TypeError."""
    with pytest.raises(ValueError, match="mode must be 'automatic' or 'requested'"):
        build_visualization_response(automatic_stage6_payload, mode="custom")

    with pytest.raises(ValueError, match="mode must be 'automatic' or 'requested'"):
        build_visualization_response(automatic_stage6_payload, mode="AUTOMATIC")

    with pytest.raises((ValueError, TypeError)):
        build_visualization_response(automatic_stage6_payload, mode=None)  # type: ignore[arg-type]


def test_19_non_mapping_payload_raises_type_error() -> None:
    """19. Non-mapping payload raises TypeError."""
    with pytest.raises(TypeError, match="visualization_payload must be a Mapping"):
        build_visualization_response(["not", "a", "dict"], mode="automatic")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="visualization_payload must be a Mapping"):
        build_visualization_response("string", mode="automatic")  # type: ignore[arg-type]


def test_20_visualization_ordering_preserved() -> None:
    """20. Ordering of visualizations is strictly preserved (no reranking)."""
    payload = {
        "status": "READY",
        "dataset_name": "test",
        "visualization_count": 3,
        "visualizations": [
            _make_count_bar_viz(score=0.999),
            _make_bar_viz(score=0.95),
            _make_count_bar_viz(score=0.88),
        ],
        "signals": [],
    }
    resp = build_visualization_response(payload, mode="automatic")
    scores = [v["score"] for v in resp["visualizations"]]
    assert scores[0] == pytest.approx(0.999)
    assert scores[1] == pytest.approx(0.95)
    assert scores[2] == pytest.approx(0.88)


def test_21_supported_modes_constant() -> None:
    """21. SUPPORTED_MODES constant is exactly {'automatic', 'requested'}."""
    assert SUPPORTED_MODES == frozenset({"automatic", "requested"})