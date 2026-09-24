"""Tests for Pipeline 3 Stage 8 — Requested Visualization Resolution.

Tests the resolve_visualization_request function across all chart types,
explicit and default aggregations, invalid/clarification request handling,
JSON safety, and determinism.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pipelines.pipeline3_results.visualization.request_parser import (
    parse_visualization_request,
)
from pipelines.pipeline3_results.visualization.request_resolver import (
    resolve_visualization_request,
)
from pipelines.pipeline3_results.visualization.request_validator import (
    validate_visualization_request,
)
from pipelines.pipeline3_results.visualization.schemas import (
    ValidatedVisualizationRequest,
    VisualizationRequest,
)


@pytest.fixture
def sample_bar_data() -> pd.DataFrame:
    """Category and measure dataset with known mathematical properties."""
    return pd.DataFrame({
        "category": ["A", "B", "A", "C", "A", "B"],
        "revenue": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "quantity": [1, 2, 3, 4, 5, 6],
    })


@pytest.fixture
def sample_temporal_data() -> pd.DataFrame:
    """Temporal and measure dataset with consecutive dates."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "date": dates,
        "sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    })


@pytest.fixture
def sample_scatter_data() -> pd.DataFrame:
    """Two numeric measure columns for scatter plotting."""
    return pd.DataFrame({
        "height": [150.0, 160.0, 170.0, 180.0, 190.0],
        "weight": [50.0, 60.0, 70.0, 80.0, 90.0],
    })


@pytest.fixture
def sample_hist_data() -> pd.DataFrame:
    """Single numeric measure column with distinct values."""
    return pd.DataFrame({
        "score": [12.0, 15.5, 22.1, 28.4, 33.0, 42.1, 55.0, 61.2, 70.0, 85.5],
    })


# ---------------------------------------------------------------------------
# A. COUNT_BAR Tests
# ---------------------------------------------------------------------------

def test_resolve_count_bar_valid(sample_bar_data: pd.DataFrame) -> None:
    """Test COUNT_BAR resolution produces exact frequencies and score is None."""
    req = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="COUNT_BAR",
        x_column="category",
        y_column=None,
        aggregation=None,
        intent="DISTRIBUTION",
        requested_columns=["category"],
    )
    result = resolve_visualization_request(sample_bar_data, req, dataset_name="test_data")

    assert result["status"] == "READY"
    assert result["dataset_name"] == "test_data"
    assert result["visualization_count"] == 1
    assert len(result["visualizations"]) == 1

    viz = result["visualizations"][0]
    assert viz["rank"] == 1
    assert viz["chart_type"] == "COUNT_BAR"
    assert viz["x_column"] == "category"
    assert viz["y_column"] is None
    assert viz["score"] is None  # Score must be None

    chart_data = viz["chart_data"]
    assert chart_data["has_data"] is True
    assert chart_data["usable_row_count"] == 6

    # Verify counts: A=3, B=2, C=1
    points = chart_data["points"]
    assert len(points) == 3
    point_dict = {p["x"]: p["y"] for p in points}
    assert point_dict == {"A": 3, "B": 2, "C": 1}
    assert point_dict["A"] == 3
    assert point_dict["B"] == 2
    assert point_dict["C"] == 1


# ---------------------------------------------------------------------------
# B. BAR Default Aggregation Tests
# ---------------------------------------------------------------------------

def test_resolve_bar_default_aggregation(sample_bar_data: pd.DataFrame) -> None:
    """Test BAR chart without explicit aggregation uses Stage 4 semantic aggregation."""
    req = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="BAR",
        x_column="category",
        y_column="revenue",
        aggregation=None,
        intent="COMPARISON",
        requested_columns=["category", "revenue"],
    )
    result = resolve_visualization_request(sample_bar_data, req)

    assert result["status"] == "READY"
    viz = result["visualizations"][0]
    assert viz["score"] is None
    assert viz["chart_type"] == "BAR"
    assert viz["x_column"] == "category"
    assert viz["y_column"] == "revenue"

    chart_data = viz["chart_data"]
    assert chart_data["has_data"] is True
    # 'revenue' has name token 'revenue' which defaults to sum
    assert chart_data["aggregation"] == "sum"
    point_dict = {p["x"]: p["y"] for p in chart_data["points"]}
    assert point_dict["A"] == 90.0  # 10 + 30 + 50
    assert point_dict["B"] == 80.0  # 20 + 60
    assert point_dict["C"] == 40.0  # 40


# ---------------------------------------------------------------------------
# C. BAR Explicit Aggregation Tests (SUM, MEAN, MEDIAN, MIN, MAX, COUNT)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("agg", "expected_a", "expected_b", "expected_c"),
    [
        ("SUM", 90.0, 80.0, 40.0),
        ("MEAN", 30.0, 40.0, 40.0),
        ("MEDIAN", 30.0, 40.0, 40.0),
        ("MIN", 10.0, 20.0, 40.0),
        ("MAX", 50.0, 60.0, 40.0),
        ("COUNT", 3.0, 2.0, 1.0),
    ],
)
def test_resolve_bar_explicit_aggregations(
    sample_bar_data: pd.DataFrame,
    agg: str,
    expected_a: float,
    expected_b: float,
    expected_c: float,
) -> None:
    """Verify that all explicit aggregations produce the exact expected numerical results."""
    req = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="BAR",
        x_column="category",
        y_column="revenue",
        aggregation=agg,  # type: ignore[arg-type]
        intent="COMPARISON",
        requested_columns=["category", "revenue"],
    )
    result = resolve_visualization_request(sample_bar_data, req)

    assert result["status"] == "READY"
    viz = result["visualizations"][0]
    assert viz["score"] is None
    chart_data = viz["chart_data"]
    assert chart_data["aggregation"] == agg.lower()

    # Verify signal records explicit aggregation
    assert any(f"explicit {agg} aggregation applied to revenue" in s for s in chart_data["signals"])

    point_dict = {p["x"]: p["y"] for p in chart_data["points"]}
    assert pytest.approx(point_dict["A"]) == expected_a
    assert pytest.approx(point_dict["B"]) == expected_b
    assert pytest.approx(point_dict["C"]) == expected_c


# ---------------------------------------------------------------------------
# D. LINE Tests
# ---------------------------------------------------------------------------

def test_resolve_line_default_aggregation(sample_temporal_data: pd.DataFrame) -> None:
    """Test LINE resolution with default aggregation preserves chronological order."""
    req = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="LINE",
        x_column="date",
        y_column="sales",
        aggregation=None,
        intent="TREND",
        requested_columns=["date", "sales"],
    )
    result = resolve_visualization_request(sample_temporal_data, req)

    assert result["status"] == "READY"
    viz = result["visualizations"][0]
    assert viz["score"] is None
    assert viz["chart_type"] == "LINE"

    chart_data = viz["chart_data"]
    assert chart_data["has_data"] is True
    points = chart_data["points"]
    assert len(points) == 10
    # Chronological check
    dates = [p["x"] for p in points]
    assert dates == sorted(dates)


@pytest.mark.parametrize("agg", ["SUM", "MEAN", "MEDIAN", "MIN", "MAX", "COUNT"])
def test_resolve_line_explicit_aggregations(sample_temporal_data: pd.DataFrame, agg: str) -> None:
    """Test LINE resolution with explicit aggregations."""
    req = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="LINE",
        x_column="date",
        y_column="sales",
        aggregation=agg,  # type: ignore[arg-type]
        intent="TREND",
        requested_columns=["date", "sales"],
    )
    result = resolve_visualization_request(sample_temporal_data, req)

    assert result["status"] == "READY"
    viz = result["visualizations"][0]
    assert viz["score"] is None
    chart_data = viz["chart_data"]
    assert chart_data["aggregation"] == agg.lower()
    assert any(f"explicit {agg} aggregation applied to sales" in s for s in chart_data["signals"])


# ---------------------------------------------------------------------------
# E. SCATTER Tests
# ---------------------------------------------------------------------------

def test_resolve_scatter_valid(sample_scatter_data: pd.DataFrame) -> None:
    """Test SCATTER resolution produces paired coordinates with aggregation None."""
    req = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="SCATTER",
        x_column="height",
        y_column="weight",
        aggregation=None,
        intent="RELATIONSHIP",
        requested_columns=["height", "weight"],
    )
    result = resolve_visualization_request(sample_scatter_data, req)

    assert result["status"] == "READY"
    viz = result["visualizations"][0]
    assert viz["score"] is None
    assert viz["chart_type"] == "SCATTER"
    chart_data = viz["chart_data"]
    assert chart_data["aggregation"] is None
    assert len(chart_data["points"]) == 5
    assert chart_data["points"][0] == {"x": 150.0, "y": 50.0}
    assert chart_data["points"][4] == {"x": 190.0, "y": 90.0}


# ---------------------------------------------------------------------------
# F. HISTOGRAM Tests
# ---------------------------------------------------------------------------

def test_resolve_histogram_valid(sample_hist_data: pd.DataFrame) -> None:
    """Test HISTOGRAM resolution produces deterministic bins and counts."""
    req = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="HISTOGRAM",
        x_column="score",
        y_column=None,
        aggregation=None,
        intent="DISTRIBUTION",
        requested_columns=["score"],
    )
    result = resolve_visualization_request(sample_hist_data, req)

    assert result["status"] == "READY"
    viz = result["visualizations"][0]
    assert viz["score"] is None
    assert viz["chart_type"] == "HISTOGRAM"
    chart_data = viz["chart_data"]
    assert chart_data["aggregation"] == "count"
    assert len(chart_data["bins"]) > 0
    assert sum(b["count"] for b in chart_data["bins"]) == 10


# ---------------------------------------------------------------------------
# G. Unsupported Aggregations Tests
# ---------------------------------------------------------------------------

def test_resolve_unsupported_aggregation_scatter(sample_scatter_data: pd.DataFrame) -> None:
    """SCATTER with aggregation returns INVALID."""
    req = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="SCATTER",
        x_column="height",
        y_column="weight",
        aggregation="MEAN",
        intent="RELATIONSHIP",
        requested_columns=["height", "weight"],
    )
    result = resolve_visualization_request(sample_scatter_data, req)

    assert result["status"] == "INVALID"
    assert result["visualization_count"] == 0
    assert result["visualizations"] == []
    assert any("does not support aggregation" in err for err in result["errors"])


def test_resolve_unsupported_aggregation_histogram(sample_hist_data: pd.DataFrame) -> None:
    """HISTOGRAM with SUM aggregation returns INVALID."""
    req = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="HISTOGRAM",
        x_column="score",
        y_column=None,
        aggregation="SUM",
        intent="DISTRIBUTION",
        requested_columns=["score"],
    )
    result = resolve_visualization_request(sample_hist_data, req)

    assert result["status"] == "INVALID"
    assert result["visualization_count"] == 0
    assert result["visualizations"] == []
    assert any("does not support aggregation" in err for err in result["errors"])


def test_resolve_unsupported_aggregation_count_bar(sample_bar_data: pd.DataFrame) -> None:
    """COUNT_BAR with MEDIAN aggregation returns INVALID."""
    req = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="COUNT_BAR",
        x_column="category",
        y_column=None,
        aggregation="MEDIAN",
        intent="DISTRIBUTION",
        requested_columns=["category"],
    )
    result = resolve_visualization_request(sample_bar_data, req)

    assert result["status"] == "INVALID"
    assert result["visualization_count"] == 0
    assert result["visualizations"] == []
    assert any("does not support aggregation" in err for err in result["errors"])


# ---------------------------------------------------------------------------
# H. NEEDS_CLARIFICATION Tests
# ---------------------------------------------------------------------------

def test_resolve_needs_clarification(sample_bar_data: pd.DataFrame) -> None:
    """NEEDS_CLARIFICATION request returns 0 visualizations and preserves options."""
    req = ValidatedVisualizationRequest(
        status="NEEDS_CLARIFICATION",
        chart_type=None,
        x_column=None,
        y_column=None,
        aggregation=None,
        intent=None,
        requested_columns=[],
        signals=["ambiguous request"],
        clarification_options=["revenue", "quantity"],
    )
    result = resolve_visualization_request(sample_bar_data, req, dataset_name="my_dataset")

    assert result["status"] == "NEEDS_CLARIFICATION"
    assert result["dataset_name"] == "my_dataset"
    assert result["visualization_count"] == 0
    assert result["visualizations"] == []
    assert result["clarification_options"] == ["revenue", "quantity"]
    assert "ambiguous request" in result["signals"]


# ---------------------------------------------------------------------------
# I. INVALID Tests
# ---------------------------------------------------------------------------

def test_resolve_invalid_request(sample_bar_data: pd.DataFrame) -> None:
    """INVALID request returns 0 visualizations and preserves errors and signals."""
    req = ValidatedVisualizationRequest(
        status="INVALID",
        chart_type="LINE",
        x_column="category",
        y_column="revenue",
        aggregation=None,
        intent="TREND",
        requested_columns=["category", "revenue"],
        signals=["rejected: category is not temporal"],
        errors=["No suitable temporal column was found."],
    )
    result = resolve_visualization_request(sample_bar_data, req)

    assert result["status"] == "INVALID"
    assert result["visualization_count"] == 0
    assert result["visualizations"] == []
    assert result["errors"] == ["No suitable temporal column was found."]
    assert "rejected: category is not temporal" in result["signals"]


# ---------------------------------------------------------------------------
# J. No Usable Data Tests
# ---------------------------------------------------------------------------

def test_resolve_empty_dataframe() -> None:
    """Empty dataframe produces NO_MEANINGFUL_VISUALIZATION."""
    empty_df = pd.DataFrame(columns=["cat", "val"])
    req = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="BAR",
        x_column="cat",
        y_column="val",
        aggregation="SUM",
        intent="COMPARISON",
        requested_columns=["cat", "val"],
    )
    result = resolve_visualization_request(empty_df, req)

    assert result["status"] == "NO_MEANINGFUL_VISUALIZATION"
    assert result["visualization_count"] == 0
    assert result["visualizations"] == []


def test_resolve_all_nan_measure() -> None:
    """All-NaN measure produces NO_MEANINGFUL_VISUALIZATION."""
    nan_df = pd.DataFrame({
        "cat": ["A", "B", "C"],
        "val": [np.nan, np.nan, np.nan],
    })
    req = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="BAR",
        x_column="cat",
        y_column="val",
        aggregation="MEAN",
        intent="COMPARISON",
        requested_columns=["cat", "val"],
    )
    result = resolve_visualization_request(nan_df, req)

    assert result["status"] == "NO_MEANINGFUL_VISUALIZATION"
    assert result["visualization_count"] == 0
    assert result["visualizations"] == []


# ---------------------------------------------------------------------------
# K. JSON Round-Trip Tests
# ---------------------------------------------------------------------------

def test_resolve_json_round_trip(sample_bar_data: pd.DataFrame) -> None:
    """Verify json.loads(json.dumps(payload)) == payload without custom encoders."""
    req = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="BAR",
        x_column="category",
        y_column="revenue",
        aggregation="MEDIAN",
        intent="COMPARISON",
        requested_columns=["category", "revenue"],
    )
    payload = resolve_visualization_request(sample_bar_data, req, dataset_name="bar_data")

    dumped = json.dumps(payload)
    loaded = json.loads(dumped)
    assert loaded == payload


# ---------------------------------------------------------------------------
# L. Determinism Tests
# ---------------------------------------------------------------------------

def test_resolve_determinism(sample_bar_data: pd.DataFrame) -> None:
    """Repeated runs on identical inputs produce identical payloads."""
    req = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="BAR",
        x_column="category",
        y_column="revenue",
        aggregation="SUM",
        intent="COMPARISON",
        requested_columns=["category", "revenue"],
    )
    run1 = resolve_visualization_request(sample_bar_data, req)
    run2 = resolve_visualization_request(sample_bar_data, req)

    assert run1 == run2


# ---------------------------------------------------------------------------
# M. Type Validation Tests
# ---------------------------------------------------------------------------

def test_resolve_type_validation(sample_bar_data: pd.DataFrame) -> None:
    """Strict TypeError validation on input parameters."""
    req = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="BAR",
        x_column="category",
        y_column="revenue",
        aggregation=None,
    )

    with pytest.raises(TypeError, match="dataframe must be a pandas DataFrame"):
        resolve_visualization_request("not_a_df", req)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="validated_request must be a ValidatedVisualizationRequest"):
        resolve_visualization_request(sample_bar_data, "not_a_req")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="pipeline1_report must be a Mapping"):
        resolve_visualization_request(sample_bar_data, req, pipeline1_report="invalid")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="dataset_name must be a string"):
        resolve_visualization_request(sample_bar_data, req, dataset_name=12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# N. Score is Explicitly None
# ---------------------------------------------------------------------------

def test_resolve_score_is_explicitly_none(sample_bar_data: pd.DataFrame) -> None:
    """Assert score is None across all chart types."""
    for chart_type, agg in [
        ("BAR", "SUM"),
        ("COUNT_BAR", None),
    ]:
        req = ValidatedVisualizationRequest(
            status="VALID",
            chart_type=chart_type,  # type: ignore[arg-type]
            x_column="category",
            y_column="revenue" if chart_type == "BAR" else None,
            aggregation=agg,  # type: ignore[arg-type]
        )
        res = resolve_visualization_request(sample_bar_data, req)
        assert res["visualizations"][0]["score"] is None


# ---------------------------------------------------------------------------
# O. Explicit Aggregation Overrides Semantic Default
# ---------------------------------------------------------------------------

def test_resolve_explicit_aggregation_overrides_semantic_default(sample_bar_data: pd.DataFrame) -> None:
    """Verify that an explicit aggregation (e.g. MEDIAN or MAX) overrides semantic default (SUM)."""
    # 'revenue' defaults to SUM
    req_default = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="BAR",
        x_column="category",
        y_column="revenue",
        aggregation=None,
    )
    res_default = resolve_visualization_request(sample_bar_data, req_default)
    assert res_default["visualizations"][0]["chart_data"]["aggregation"] == "sum"

    # Explicit MEDIAN override
    req_median = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="BAR",
        x_column="category",
        y_column="revenue",
        aggregation="MEDIAN",
    )
    res_median = resolve_visualization_request(sample_bar_data, req_median)
    assert res_median["visualizations"][0]["chart_data"]["aggregation"] == "median"

    # Explicit MAX override
    req_max = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="BAR",
        x_column="category",
        y_column="revenue",
        aggregation="MAX",
    )
    res_max = resolve_visualization_request(sample_bar_data, req_max)
    assert res_max["visualizations"][0]["chart_data"]["aggregation"] == "max"
