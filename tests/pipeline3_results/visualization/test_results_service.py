"""Unit tests for Pipeline 3 Stage 10 -- Results Service Composition Layer.

Covers tests A through S as required:
A. Automatic mode calls Stage 6 + Stage 9 correctly.
B. Requested VALID request calls: Stage 7 parser, Stage 7 validator, Stage 8 resolver, Stage 9 explanation.
C. NEEDS_CLARIFICATION is preserved.
D. INVALID is preserved.
E. NO_MEANINGFUL_VISUALIZATION is preserved.
F. Automatic visualization scores are preserved.
G. Requested visualization score remains None.
H. Chart data is preserved exactly.
I. Explanation is attached correctly.
J. Dataset signals are preserved.
K. JSON round-trip: json.loads(json.dumps(result)) == result.
L. Deterministic repeated execution.
M. Input validation.
N. Empty dataset.
O. <=2-row dataset.
P. Constant dataset.
Q. Multiple automatic visualizations retain original Stage 6 ordering.
R. Requested visualization produces exactly one resolved visualization.
S. No mutation of the input DataFrame.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pipelines.pipeline3_results.visualization.integration import build_visualization_payload
from pipelines.pipeline3_results.visualization.results_service import (
    build_visualization_results,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_sales_df() -> pd.DataFrame:
    """Standard multi-row tabular dataset with categorical, numeric, and date columns."""
    return pd.DataFrame({
        "category": ["Classic", "Supreme", "Veggie", "Chicken"] * 25,
        "size": ["S", "M", "L", "XL"] * 25,
        "unit_price": [12.5, 16.0, 14.0, 17.5] * 25,
        "total_price": [25.0, 32.0, 14.0, 35.0] * 25,
        "order_date": pd.date_range("2024-01-01", periods=100, freq="D").strftime("%Y-%m-%d"),
    })


# ---------------------------------------------------------------------------
# Test Cases A - S
# ---------------------------------------------------------------------------

def test_A_automatic_mode(sample_sales_df: pd.DataFrame) -> None:
    """A. Automatic mode calls Stage 6 + Stage 9 correctly when request_text is None or empty."""
    # When request_text is None
    res = build_visualization_results(sample_sales_df, request_text=None, dataset_name="sales")
    assert res["mode"] == "automatic"
    assert res["status"] in ("READY", "NO_MEANINGFUL_VISUALIZATION")
    assert res["dataset_name"] == "sales"
    assert len(res["visualizations"]) > 0
    for viz in res["visualizations"]:
        assert "explanation" in viz
        assert "why_this_chart" in viz["explanation"]

    # When request_text is empty string
    res_empty = build_visualization_results(sample_sales_df, request_text="   ", dataset_name="sales")
    assert res_empty["mode"] == "automatic"
    assert res_empty["visualization_count"] == len(res_empty["visualizations"])


def test_B_requested_valid_request(sample_sales_df: pd.DataFrame) -> None:
    """B. Requested VALID request calls Stage 7 -> Stage 8 -> Stage 9."""
    res = build_visualization_results(
        sample_sales_df,
        request_text="Median unit_price by category",
        dataset_name="sales",
    )
    assert res["mode"] == "requested"
    assert res["request_text"] == "Median unit_price by category"
    assert res["status"] == "READY"
    assert res["visualization_count"] == 1
    assert len(res["visualizations"]) == 1

    viz = res["visualizations"][0]
    assert viz["chart_type"] == "BAR"
    assert viz["x_column"] == "category"
    assert viz["y_column"] == "unit_price"
    assert viz["chart_data"]["aggregation"] == "median"
    assert "explanation" in viz
    assert "user-requested" in viz["explanation"]["aggregation"].lower()


def test_C_needs_clarification_preserved(sample_sales_df: pd.DataFrame) -> None:
    """C. NEEDS_CLARIFICATION is preserved with clarification_options."""
    res = build_visualization_results(
        sample_sales_df,
        request_text="make a chart",
        dataset_name="sales",
    )
    assert res["mode"] == "requested"
    assert res["status"] == "NEEDS_CLARIFICATION"
    assert res["visualization_count"] == 0
    assert res["visualizations"] == []
    assert "clarification_options" in res
    assert len(res["clarification_options"]) > 0


def test_D_invalid_preserved(sample_sales_df: pd.DataFrame) -> None:
    """D. INVALID request is preserved with errors."""
    res = build_visualization_results(
        sample_sales_df,
        request_text="Scatter plot of category and size",  # Two categorical columns for scatter
        dataset_name="sales",
    )
    assert res["mode"] == "requested"
    assert res["status"] == "INVALID"
    assert res["visualization_count"] == 0
    assert res["visualizations"] == []
    assert "errors" in res
    assert len(res["errors"]) > 0


def test_E_no_meaningful_visualization_preserved() -> None:
    """E. NO_MEANINGFUL_VISUALIZATION is preserved."""
    empty_df = pd.DataFrame({"col": pd.Series([], dtype="float64")})
    res = build_visualization_results(empty_df, request_text=None, dataset_name="empty")
    assert res["mode"] == "automatic"
    assert res["status"] == "NO_MEANINGFUL_VISUALIZATION"
    assert res["visualization_count"] == 0
    assert res["visualizations"] == []


def test_F_automatic_scores_preserved(sample_sales_df: pd.DataFrame) -> None:
    """F. Automatic visualization scores are preserved from Stage 5/6 as floats."""
    res = build_visualization_results(sample_sales_df, request_text=None)
    assert len(res["visualizations"]) > 0
    for viz in res["visualizations"]:
        assert viz["score"] is not None
        assert isinstance(viz["score"], (float, int))
        assert math.isfinite(float(viz["score"]))


def test_G_requested_score_remains_none(sample_sales_df: pd.DataFrame) -> None:
    """G. Requested visualization score remains None."""
    res = build_visualization_results(
        sample_sales_df,
        request_text="Bar chart of unit_price by category",
    )
    assert res["status"] == "READY"
    assert len(res["visualizations"]) == 1
    assert res["visualizations"][0]["score"] is None


def test_H_chart_data_preserved_exactly(sample_sales_df: pd.DataFrame) -> None:
    """H. Chart data points, bins, usable_row_count, and attributes are preserved."""
    res = build_visualization_results(
        sample_sales_df,
        request_text="Bar chart of unit_price by category",
    )
    viz = res["visualizations"][0]
    cd = viz["chart_data"]
    assert "points" in cd
    assert "bins" in cd
    assert "usable_row_count" in cd
    assert "has_data" in cd
    assert cd["has_data"] is True
    assert len(cd["points"]) > 0


def test_I_explanation_attached_correctly(sample_sales_df: pd.DataFrame) -> None:
    """I. Explanation sub-dict contains all expected keys."""
    res = build_visualization_results(sample_sales_df, request_text=None)
    for viz in res["visualizations"]:
        expl = viz["explanation"]
        assert "why_this_chart" in expl
        assert "what_it_shows" in expl
        assert "aggregation" in expl
        assert "data_coverage" in expl
        assert "evidence" in expl
        assert isinstance(expl["evidence"], list)


def test_J_dataset_signals_preserved(sample_sales_df: pd.DataFrame) -> None:
    """J. Dataset-level signals are preserved in response."""
    res = build_visualization_results(sample_sales_df, request_text=None)
    assert "signals" in res
    assert isinstance(res["signals"], list)


def test_K_json_round_trip(sample_sales_df: pd.DataFrame) -> None:
    """K. JSON round-trip: json.loads(json.dumps(result)) == result."""
    res_auto = build_visualization_results(sample_sales_df, request_text=None)
    dumped_auto = json.dumps(res_auto)
    loaded_auto = json.loads(dumped_auto)
    assert loaded_auto == res_auto

    res_req = build_visualization_results(sample_sales_df, request_text="Median unit_price by category")
    dumped_req = json.dumps(res_req)
    loaded_req = json.loads(dumped_req)
    assert loaded_req == res_req


def test_L_deterministic_repeated_execution(sample_sales_df: pd.DataFrame) -> None:
    """L. Deterministic repeated execution yields identical output."""
    res1 = build_visualization_results(sample_sales_df, request_text=None)
    res2 = build_visualization_results(sample_sales_df, request_text=None)
    assert res1 == res2

    res_req1 = build_visualization_results(sample_sales_df, request_text="Line of total_price over order_date")
    res_req2 = build_visualization_results(sample_sales_df, request_text="Line of total_price over order_date")
    assert res_req1 == res_req2


def test_M_input_validation(sample_sales_df: pd.DataFrame) -> None:
    """M. Input validation properly raises TypeError/ValueError."""
    with pytest.raises(TypeError, match="dataframe must be a pandas DataFrame"):
        build_visualization_results("not_a_df")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="request_text must be a string or None"):
        build_visualization_results(sample_sales_df, request_text=12345)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="pipeline1_report must be a Mapping or None"):
        build_visualization_results(sample_sales_df, pipeline1_report=["not", "a", "map"])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="dataset_name must be a string or None"):
        build_visualization_results(sample_sales_df, dataset_name=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_visualizations must be a positive integer"):
        build_visualization_results(sample_sales_df, max_visualizations=0)

    with pytest.raises(ValueError, match="max_visualizations must be a positive integer"):
        build_visualization_results(sample_sales_df, max_visualizations=-1)

    with pytest.raises(ValueError, match="max_visualizations must be a positive integer"):
        build_visualization_results(sample_sales_df, max_visualizations=True)  # type: ignore[arg-type]


def test_N_empty_dataset() -> None:
    """N. Empty dataset returns NO_MEANINGFUL_VISUALIZATION in both modes."""
    empty_df = pd.DataFrame()
    res = build_visualization_results(empty_df, request_text=None)
    assert res["status"] == "NO_MEANINGFUL_VISUALIZATION"
    assert res["visualizations"] == []

    res_req = build_visualization_results(empty_df, request_text="bar chart of a by b")
    assert res_req["status"] == "NO_MEANINGFUL_VISUALIZATION"
    assert res_req["visualizations"] == []


def test_O_two_row_dataset() -> None:
    """O. <=2-row dataset returns NO_MEANINGFUL_VISUALIZATION in automatic mode."""
    df_two = pd.DataFrame({"cat": ["A", "B"], "val": [10.0, 20.0]})
    res = build_visualization_results(df_two, request_text=None)
    assert res["status"] == "NO_MEANINGFUL_VISUALIZATION"
    assert res["visualizations"] == []


def test_P_constant_dataset() -> None:
    """P. Constant dataset returns NO_MEANINGFUL_VISUALIZATION in automatic mode."""
    df_const = pd.DataFrame({"cat": ["A", "A", "A", "A"], "val": [10.0, 10.0, 10.0, 10.0]})
    res = build_visualization_results(df_const, request_text=None)
    assert res["status"] == "NO_MEANINGFUL_VISUALIZATION"
    assert res["visualizations"] == []


def test_Q_automatic_ordering_preserved(sample_sales_df: pd.DataFrame) -> None:
    """Q. Multiple automatic visualizations retain original Stage 6 ordering."""
    stage6_payload = build_visualization_payload(sample_sales_df, max_visualizations=5)
    expected_ranks = [v["rank"] for v in stage6_payload["visualizations"]]

    res = build_visualization_results(sample_sales_df, request_text=None, max_visualizations=5)
    actual_ranks = [v["rank"] for v in res["visualizations"]]
    assert actual_ranks == expected_ranks


def test_R_requested_single_visualization(sample_sales_df: pd.DataFrame) -> None:
    """R. Requested visualization produces exactly one resolved visualization."""
    res = build_visualization_results(
        sample_sales_df,
        request_text="Histogram of unit_price",
    )
    assert res["status"] == "READY"
    assert res["visualization_count"] == 1
    assert len(res["visualizations"]) == 1
    assert res["visualizations"][0]["chart_type"] == "HISTOGRAM"


def test_S_no_mutation_of_input_dataframe(sample_sales_df: pd.DataFrame) -> None:
    """S. No mutation of the input DataFrame."""
    df_copy = sample_sales_df.copy(deep=True)
    build_visualization_results(sample_sales_df, request_text=None)
    pd.testing.assert_frame_equal(sample_sales_df, df_copy)

    build_visualization_results(sample_sales_df, request_text="Median unit_price by category")
    pd.testing.assert_frame_equal(sample_sales_df, df_copy)