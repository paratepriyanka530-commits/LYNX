"""Unit tests for Pipeline 3 Stage 9 — Visualization Explanation & XAI Evidence.

Verifies deterministic, factual explanation generation across all chart types,
aggregations, attribution logic, evidence statements, edge cases, forbidden
language checks, and JSON safety.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

import pytest

from pipelines.pipeline3_results.visualization.explanation import (
    FORBIDDEN_WORDS,
    build_visualization_explanation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def count_bar_item() -> dict[str, Any]:
    return {
        "rank": 1,
        "chart_type": "COUNT_BAR",
        "x_column": "pizza_category",
        "y_column": None,
        "score": None,
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


@pytest.fixture
def bar_item() -> dict[str, Any]:
    return {
        "rank": 1,
        "chart_type": "BAR",
        "x_column": "pizza_category",
        "y_column": "total_price",
        "score": 0.85,
        "chart_data": {
            "chart_type": "BAR",
            "x_column": "pizza_category",
            "y_column": "total_price",
            "aggregation": "sum",
            "points": [
                {"x": "Classic", "y": 220053.1},
                {"x": "Supreme", "y": 208197.0},
                {"x": "Chicken", "y": 195919.5},
                {"x": "Veggie", "y": 193690.4},
            ],
            "bins": [],
            "signals": [
                "total_price aggregated using sum (Pipeline 1 semantic: currency)",
                "4 category groups produced",
            ],
            "usable_row_count": 48620,
            "sampled": False,
            "has_data": True,
        },
        "signals": ["BAR candidate ranked #1"],
    }


@pytest.fixture
def line_item() -> dict[str, Any]:
    return {
        "rank": 1,
        "chart_type": "LINE",
        "x_column": "order_date",
        "y_column": "total_price",
        "score": None,
        "chart_data": {
            "chart_type": "LINE",
            "x_column": "order_date",
            "y_column": "total_price",
            "aggregation": "sum",
            "points": [
                {"x": "2024-01-01", "y": 1250.0},
                {"x": "2024-06-01", "y": 1400.0},
                {"x": "2024-12-31", "y": 1780.0},
            ],
            "bins": [],
            "signals": [
                "explicit SUM aggregation applied to total_price",
                "resampled to weekly (53 periods)",
            ],
            "usable_row_count": 48620,
            "sampled": False,
            "has_data": True,
        },
        "signals": ["resolved requested visualization: LINE(x=order_date, y=total_price)"],
    }


@pytest.fixture
def scatter_item() -> dict[str, Any]:
    return {
        "rank": 1,
        "chart_type": "SCATTER",
        "x_column": "unit_price",
        "y_column": "total_price",
        "score": 0.75,
        "chart_data": {
            "chart_type": "SCATTER",
            "x_column": "unit_price",
            "y_column": "total_price",
            "aggregation": None,
            "points": [
                {"x": 12.5, "y": 12.5},
                {"x": 16.0, "y": 32.0},
                {"x": 20.75, "y": 41.5},
            ],
            "bins": [],
            "signals": ["3 valid observations used"],
            "usable_row_count": 3,
            "sampled": False,
            "has_data": True,
        },
        "signals": ["SCATTER candidate ranked"],
    }


@pytest.fixture
def hist_item() -> dict[str, Any]:
    return {
        "rank": 1,
        "chart_type": "HISTOGRAM",
        "x_column": "frequency",
        "y_column": None,
        "score": 0.65,
        "chart_data": {
            "chart_type": "HISTOGRAM",
            "x_column": "frequency",
            "y_column": None,
            "aggregation": "count",
            "points": [],
            "bins": [
                {"bin_start": 200.0, "bin_end": 1000.0, "count": 520},
                {"bin_start": 1000.0, "bin_end": 2000.0, "count": 680},
                {"bin_start": 2000.0, "bin_end": 5000.0, "count": 303},
            ],
            "signals": ["histogram generated using 3 bins"],
            "usable_row_count": 1503,
            "sampled": False,
            "has_data": True,
        },
        "signals": ["HISTOGRAM candidate ranked"],
    }


# ---------------------------------------------------------------------------
# A–E: Chart Type Specific Explanations
# ---------------------------------------------------------------------------

def test_count_bar_explanation(count_bar_item: dict[str, Any]) -> None:
    res = build_visualization_explanation(count_bar_item)
    assert res["chart_type"] == "COUNT_BAR"
    assert res["x_column"] == "pizza_category"
    assert res["y_column"] is None
    exp = res["explanation"]
    assert exp["why_this_chart"] == "Shows the distribution of observations across categories of 'pizza_category'."
    assert exp["what_it_shows"] == "Displays 4 categories using observation counts."
    assert "4 displayed categories from 48,620 usable rows." in exp["data_coverage"]


def test_bar_explanation(bar_item: dict[str, Any]) -> None:
    res = build_visualization_explanation(bar_item)
    assert res["chart_type"] == "BAR"
    assert res["x_column"] == "pizza_category"
    assert res["y_column"] == "total_price"
    exp = res["explanation"]
    assert exp["why_this_chart"] == "Shows a comparison of 'total_price' across the categories of 'pizza_category'."
    assert exp["what_it_shows"] == "Displays 4 categories using SUM aggregation."
    assert "4 displayed categories from 48,620 usable rows." in exp["data_coverage"]


def test_line_explanation(line_item: dict[str, Any]) -> None:
    res = build_visualization_explanation(line_item)
    assert res["chart_type"] == "LINE"
    assert res["x_column"] == "order_date"
    assert res["y_column"] == "total_price"
    exp = res["explanation"]
    assert exp["why_this_chart"] == "Shows how 'total_price' changes over the temporal dimension 'order_date'."
    assert exp["what_it_shows"] == "Displays 3 temporal points in chronological order."


def test_scatter_explanation(scatter_item: dict[str, Any]) -> None:
    res = build_visualization_explanation(scatter_item)
    assert res["chart_type"] == "SCATTER"
    assert res["x_column"] == "unit_price"
    assert res["y_column"] == "total_price"
    exp = res["explanation"]
    assert exp["why_this_chart"] == "Shows the relationship between the numeric measures 'unit_price' and 'total_price'."
    assert exp["what_it_shows"] == "Displays 3 paired observations."


def test_histogram_explanation(hist_item: dict[str, Any]) -> None:
    res = build_visualization_explanation(hist_item)
    assert res["chart_type"] == "HISTOGRAM"
    assert res["x_column"] == "frequency"
    assert res["y_column"] is None
    exp = res["explanation"]
    assert exp["why_this_chart"] == "Shows the distribution of values in the numeric measure 'frequency'."
    assert exp["what_it_shows"] == "Displays 3 bins representing 1,503 observations."


# ---------------------------------------------------------------------------
# F–L: Aggregations (SUM, MEAN, MEDIAN, MIN, MAX, COUNT, None)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("agg", "expected_text"),
    [
        ("sum", "Displays 4 categories using SUM aggregation."),
        ("mean", "Displays 4 categories using MEAN aggregation."),
        ("median", "Displays 4 categories using MEDIAN aggregation."),
        ("min", "Displays 4 categories using MIN aggregation."),
        ("max", "Displays 4 categories using MAX aggregation."),
        ("count", "Displays 4 categories using COUNT aggregation."),
    ],
)
def test_aggregations_bar(bar_item: dict[str, Any], agg: str, expected_text: str) -> None:
    item = copy.deepcopy(bar_item)
    item["chart_data"]["aggregation"] = agg
    res = build_visualization_explanation(item)
    assert res["explanation"]["what_it_shows"] == expected_text


def test_none_aggregation(scatter_item: dict[str, Any]) -> None:
    res = build_visualization_explanation(scatter_item)
    assert res["explanation"]["aggregation"] == "None / no explicit aggregation applied."


# ---------------------------------------------------------------------------
# M–O: Aggregation Attribution
# ---------------------------------------------------------------------------

def test_user_requested_aggregation_attribution(bar_item: dict[str, Any]) -> None:
    item = copy.deepcopy(bar_item)
    item["chart_data"]["aggregation"] = "median"
    item["chart_data"]["signals"] = ["explicit MEDIAN aggregation applied to unit_price"]
    res = build_visualization_explanation(item)
    assert res["explanation"]["aggregation"] == "User-requested MEDIAN aggregation was applied."


def test_automatic_semantic_aggregation_attribution(bar_item: dict[str, Any]) -> None:
    item = copy.deepcopy(bar_item)
    item["chart_data"]["aggregation"] = "sum"
    item["chart_data"]["signals"] = ["total_price aggregated using sum (Pipeline 1 semantic: currency)"]
    res = build_visualization_explanation(item)
    assert res["explanation"]["aggregation"] == "Semantic/default aggregation was applied."


def test_unknown_aggregation_attribution(bar_item: dict[str, Any]) -> None:
    item = copy.deepcopy(bar_item)
    item["score"] = None
    item["signals"] = []
    item["chart_data"]["aggregation"] = "sum"
    item["chart_data"]["signals"] = []
    res = build_visualization_explanation(item)
    assert res["explanation"]["aggregation"] == "Aggregation attribution is unavailable."


# ---------------------------------------------------------------------------
# P–Q: Category Evidence & Ties
# ---------------------------------------------------------------------------

def test_category_evidence(bar_item: dict[str, Any]) -> None:
    res = build_visualization_explanation(bar_item)
    ev = res["explanation"]["evidence"]
    assert "4 categories are displayed." in ev
    assert any("Highest displayed value: Classic = 220,053.1." in s for s in ev)
    assert any("Lowest displayed value: Veggie = 193,690.4." in s for s in ev)


def test_category_evidence_highest_tie() -> None:
    item = {
        "chart_type": "BAR",
        "x_column": "group",
        "y_column": "val",
        "chart_data": {
            "chart_type": "BAR",
            "x_column": "group",
            "y_column": "val",
            "points": [
                {"x": "A", "y": 10},
                {"x": "B", "y": 10},
                {"x": "C", "y": 5},
            ],
            "usable_row_count": 3,
            "has_data": True,
        },
    }
    res = build_visualization_explanation(item)
    ev = res["explanation"]["evidence"]
    assert any("Highest displayed value is tied: A = 10 and B = 10." in s for s in ev)
    assert any("Lowest displayed value: C = 5." in s for s in ev)


def test_category_evidence_all_equal() -> None:
    item = {
        "chart_type": "BAR",
        "x_column": "group",
        "y_column": "val",
        "chart_data": {
            "chart_type": "BAR",
            "x_column": "group",
            "y_column": "val",
            "points": [
                {"x": "A", "y": 7},
                {"x": "B", "y": 7},
            ],
            "usable_row_count": 2,
            "has_data": True,
        },
    }
    res = build_visualization_explanation(item)
    ev = res["explanation"]["evidence"]
    assert any("All displayed categories have equal value: 7." in s for s in ev)


# ---------------------------------------------------------------------------
# R–U: Line Endpoint Evidence
# ---------------------------------------------------------------------------

def test_line_endpoint_evidence_increased(line_item: dict[str, Any]) -> None:
    res = build_visualization_explanation(line_item)
    ev = res["explanation"]["evidence"]
    assert "Earliest displayed point: 2024-01-01 = 1,250." in ev
    assert "Latest displayed point: 2024-12-31 = 1,780." in ev
    assert "The displayed endpoint value increased from 1,250 to 1,780." in ev
    assert "3 temporal points are displayed." in ev


def test_line_endpoint_evidence_decreased() -> None:
    item = {
        "chart_type": "LINE",
        "x_column": "date",
        "y_column": "temp",
        "chart_data": {
            "chart_type": "LINE",
            "x_column": "date",
            "y_column": "temp",
            "points": [
                {"x": "2024-01-01", "y": 100},
                {"x": "2024-01-02", "y": 50},
            ],
            "has_data": True,
        },
    }
    res = build_visualization_explanation(item)
    ev = res["explanation"]["evidence"]
    assert "The displayed endpoint value decreased from 100 to 50." in ev


def test_line_endpoint_evidence_equal() -> None:
    item = {
        "chart_type": "LINE",
        "x_column": "date",
        "y_column": "temp",
        "chart_data": {
            "chart_type": "LINE",
            "x_column": "date",
            "y_column": "temp",
            "points": [
                {"x": "2024-01-01", "y": 25},
                {"x": "2024-01-02", "y": 25},
            ],
            "has_data": True,
        },
    }
    res = build_visualization_explanation(item)
    ev = res["explanation"]["evidence"]
    assert "The displayed endpoint value stayed equal from 25 to 25." in ev


# ---------------------------------------------------------------------------
# V–W: Scatter Sampling & Histogram Evidence
# ---------------------------------------------------------------------------

def test_scatter_sampling_evidence() -> None:
    item = {
        "chart_type": "SCATTER",
        "x_column": "x",
        "y_column": "y",
        "chart_data": {
            "chart_type": "SCATTER",
            "x_column": "x",
            "y_column": "y",
            "points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
            "sampled": True,
            "has_data": True,
        },
    }
    res = build_visualization_explanation(item)
    ev = res["explanation"]["evidence"]
    assert "2 paired observations are displayed after deterministic sampling." in ev
    assert "Data was deterministically sampled to satisfy the visualization point limit." in res["explanation"]["data_coverage"]


def test_histogram_evidence(hist_item: dict[str, Any]) -> None:
    res = build_visualization_explanation(hist_item)
    ev = res["explanation"]["evidence"]
    assert "3 bins are displayed." in ev
    assert any("Highest-frequency bin: [1,000, 2,000) = 680 observations." in s for s in ev)
    assert any("Lowest-frequency bin: [2,000, 5,000) = 303 observations." in s for s in ev)


# ---------------------------------------------------------------------------
# X–AA: Missing / Empty / Invalid / Clarification Handling
# ---------------------------------------------------------------------------

def test_missing_chart_data() -> None:
    item = {"chart_type": "BAR", "x_column": "a", "y_column": "b"}
    res = build_visualization_explanation(item)
    assert res["explanation"]["why_this_chart"] == "Visualization rationale is unavailable."
    assert res["explanation"]["what_it_shows"] == "No displayable chart data available."
    assert res["explanation"]["evidence"] == []


def test_empty_payload() -> None:
    res = build_visualization_explanation({})
    assert res["chart_type"] == "UNKNOWN"
    assert res["explanation"]["what_it_shows"] == "No displayable chart data available."
    assert res["explanation"]["evidence"] == []


def test_invalid_payload() -> None:
    payload = {
        "status": "INVALID",
        "visualization_count": 0,
        "visualizations": [],
        "errors": ["Rejected: column is not numeric"],
        "signals": ["rejected"],
    }
    res = build_visualization_explanation(payload)
    assert res["explanation"]["why_this_chart"] == "Visualization rationale is unavailable."
    assert res["explanation"]["what_it_shows"] == "No displayable chart data available."
    assert res["explanation"]["evidence"] == []


def test_needs_clarification_payload() -> None:
    payload = {
        "status": "NEEDS_CLARIFICATION",
        "visualization_count": 0,
        "visualizations": [],
        "clarification_options": ["col1", "col2"],
        "signals": ["ambiguous"],
    }
    res = build_visualization_explanation(payload)
    assert res["explanation"]["why_this_chart"] == "Visualization rationale is unavailable."
    assert res["explanation"]["what_it_shows"] == "No displayable chart data available."
    assert res["explanation"]["evidence"] == []


# ---------------------------------------------------------------------------
# AB–AG: Contract Invariants (JSON, Determinism, Safety, Immutability)
# ---------------------------------------------------------------------------

def test_json_round_trip(bar_item: dict[str, Any]) -> None:
    res = build_visualization_explanation(bar_item)
    dumped = json.dumps(res)
    loaded = json.loads(dumped)
    assert loaded == res


def test_deterministic_repeated_execution(line_item: dict[str, Any]) -> None:
    res1 = build_visualization_explanation(line_item)
    res2 = build_visualization_explanation(line_item)
    assert res1 == res2


def test_forbidden_language_scan(
    count_bar_item: dict[str, Any],
    bar_item: dict[str, Any],
    line_item: dict[str, Any],
    scatter_item: dict[str, Any],
    hist_item: dict[str, Any],
) -> None:
    for item in [count_bar_item, bar_item, line_item, scatter_item, hist_item]:
        res = build_visualization_explanation(item)
        all_text = " ".join([
            res["explanation"]["why_this_chart"],
            res["explanation"]["what_it_shows"],
            res["explanation"]["aggregation"],
            res["explanation"]["data_coverage"],
            " ".join(res["explanation"]["evidence"]),
            " ".join(res["signals"]),
        ])
        words = re.findall(r"\b[a-zA-Z]+\b", all_text.lower())
        for w in words:
            assert w not in FORBIDDEN_WORDS, f"Forbidden word '{w}' found in: {all_text}"


def test_malformed_signals(bar_item: dict[str, Any]) -> None:
    item = copy.deepcopy(bar_item)
    item["signals"] = [123, None, "valid signal"]  # type: ignore[list-item]
    res = build_visualization_explanation(item, dataset_signals=["another", None])  # type: ignore[list-item]
    assert res["explanation"]["what_it_shows"] == "Displays 4 categories using SUM aggregation."


def test_json_primitive_purity(bar_item: dict[str, Any]) -> None:
    res = build_visualization_explanation(bar_item)
    # Check that all dictionary values and nested collections consist of primitive types
    def check_primitives(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert isinstance(k, str)
                check_primitives(v)
        elif isinstance(obj, list):
            for v in obj:
                check_primitives(v)
        else:
            assert isinstance(obj, (str, int, float, bool)) or obj is None

    check_primitives(res)


def test_input_visualization_not_mutated(bar_item: dict[str, Any]) -> None:
    orig = copy.deepcopy(bar_item)
    _ = build_visualization_explanation(bar_item)
    assert bar_item == orig

