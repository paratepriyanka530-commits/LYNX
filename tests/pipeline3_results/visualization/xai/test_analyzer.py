from __future__ import annotations

import math

import pandas as pd
import pytest

from pipelines.pipeline3_results.execution.pandas_executor import PandasAnalyticalExecutor
from pipelines.pipeline3_results.grounding.dataset_knowledge import build_dataset_knowledge
from pipelines.pipeline3_results.grounding.schemas import AnalyticalExpression
from pipelines.pipeline3_results.insights.landing.opportunities import make_plan
from pipelines.pipeline3_results.visualization.intelligence.schemas import ChartFamily, FieldEncoding
from pipelines.pipeline3_results.visualization.intelligence.service import build_requested_visualization
from pipelines.pipeline3_results.visualization.xai.analyzer import analyze_chart, select_prose_facts
from pipelines.pipeline3_results.visualization.xai.service import VisualizationXAIService


def _build(frame: pd.DataFrame, plan, family: ChartFamily | None = None):
    knowledge = build_dataset_knowledge(None, dataset_id="test", dataframe_schema={str(c): str(t) for c, t in frame.dtypes.items()})
    result = PandasAnalyticalExecutor().execute(plan, frame, knowledge)
    _, visual = build_requested_visualization(plan, result, knowledge, family)
    assert visual is not None
    return result, visual


def _facts(result, visual):
    facts, limitations = analyze_chart(result, visual.plan, visual.chart_spec)
    return {item.fact_type: item for item in facts}, facts, limitations


def test_bar_facts_cover_rank_extremes_difference_share_and_concentration() -> None:
    frame = pd.DataFrame({"category": ["A", "B", "C"], "amount": [10.0, 20.0, 70.0]})
    plan = make_plan("GROUP_AGGREGATE", measures=[AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")], dimension="category")
    result, visual = _build(frame, plan, ChartFamily.BAR)
    by_type, facts, _ = _facts(result, visual)
    assert by_type["MAXIMUM"].values["category"] == "C"
    assert by_type["MINIMUM"].values["category"] == "A"
    assert by_type["RANKING"].values["order"] == ["C", "B", "A"]
    assert by_type["ABSOLUTE_CHANGE"].values["difference"] == 60.0
    assert by_type["PERCENT_CHANGE"].values["percent"] == 600.0
    assert by_type["TOP_SHARE"].values["share_percent"] == 70.0
    assert by_type["CONCENTRATION"].values["share_percent"] == 100.0
    assert 2 <= len(select_prose_facts(facts)) <= 4


def test_bar_ties_and_coverage_caveat() -> None:
    frame = pd.DataFrame({"category": ["A", "B", "C", "D"], "amount": [10.0, 10.0, None, None]})
    plan = make_plan("GROUP_AGGREGATE", measures=[AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")], dimension="category")
    result, visual = _build(frame, plan, ChartFamily.BAR)
    by_type, _, limitations = _facts(result, visual)
    assert "TIE" in by_type
    assert "COVERAGE" in by_type
    assert any("excluded" in item for item in limitations)


@pytest.mark.parametrize(("values", "direction", "change"), [
    ([10.0, 15.0, 13.0, 20.0], "increased", 10.0),
    ([20.0, 15.0, 15.0, 10.0], "decreased", -10.0),
    ([10.0, 10.0, 10.0, 10.0], "flat", 0.0),
])
def test_line_change_direction_extremes_and_adjacent_moves(values, direction, change) -> None:
    frame = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=4, freq="YS"), "amount": values})
    plan = make_plan("TREND", measures=[AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")], temporal="date")
    result, visual = _build(frame, plan, ChartFamily.LINE)
    by_type, _, _ = _facts(result, visual)
    assert by_type["ABSOLUTE_CHANGE"].values["change"] == change
    assert by_type["TREND_DIRECTION"].values["direction"] == direction
    assert {"MAXIMUM", "MINIMUM", "LARGEST_INCREASE", "LARGEST_DECREASE"} <= set(by_type)


def test_line_invalid_percent_denominator_is_explicit() -> None:
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="YS"), "amount": [0.0, 2.0, 4.0]})
    plan = make_plan("TREND", measures=[AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")], temporal="date")
    result, visual = _build(frame, plan, ChartFamily.LINE)
    by_type, _, limitations = _facts(result, visual)
    assert "PERCENT_CHANGE" not in by_type
    assert any("first value is zero" in item for item in limitations)


def test_pie_shares_and_equal_share_tie() -> None:
    frame = pd.DataFrame({"category": ["A", "B", "C"], "amount": [5.0, 5.0, 10.0]})
    plan = make_plan("PART_TO_WHOLE", measures=[AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")], dimension="category")
    result, visual = _build(frame, plan, ChartFamily.DOUGHNUT)
    by_type, _, _ = _facts(result, visual)
    assert by_type["TOP_SHARE"].values["share_percent"] == 50.0
    assert "TIE" in by_type
    assert by_type["CONCENTRATION"].values["category_count"] == 3


def test_invalid_part_to_whole_state_is_controlled() -> None:
    frame = pd.DataFrame({"category": ["A", "B"], "amount": [1.0, 2.0]})
    plan = make_plan("PART_TO_WHOLE", measures=[AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")], dimension="category")
    result, visual = _build(frame, plan, ChartFamily.PIE)
    invalid = result.model_copy(update={"rows": [{"category": "A", "value": -1.0}, {"category": "B", "value": 1.0}]})
    facts, limitations = analyze_chart(invalid, visual.plan, visual.chart_spec)
    assert not [item for item in facts if item.fact_type == "TOP_SHARE"]
    assert any("positive displayed total" in item for item in limitations)


@pytest.mark.parametrize(("ys", "direction", "strength"), [
    ([2, 4, 6, 8, 10], "positive", "very strong"),
    ([10, 8, 6, 4, 2], "negative", "very strong"),
    ([2, 5, 1, 4, 3], "positive", "very weak"),
])
def test_scatter_pearson_is_calculated_and_noncausal(ys, direction, strength) -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": ys})
    plan = make_plan("RELATIONSHIP", measures=[AnalyticalExpression(kind="COLUMN", column="x"), AnalyticalExpression(kind="COLUMN", column="y")])
    result, visual = _build(frame, plan, ChartFamily.SCATTER)
    by_type, _, limitations = _facts(result, visual)
    assert by_type["PAIR_COUNT"].values["pair_count"] == 5
    correlation = by_type["CORRELATION"].values
    assert correlation["direction"] == direction
    assert correlation["strength"] == strength
    assert math.isfinite(correlation["coefficient"])
    assert "Correlation does not establish causation." in limitations


def test_scatter_constant_input_does_not_invent_correlation() -> None:
    frame = pd.DataFrame({"x": [1, 1, 1], "y": [1, 2, 3]})
    plan = make_plan("RELATIONSHIP", measures=[AnalyticalExpression(kind="COLUMN", column="x"), AnalyticalExpression(kind="COLUMN", column="y")])
    result, visual = _build(frame, plan, ChartFamily.SCATTER)
    by_type, _, limitations = _facts(result, visual)
    assert "CORRELATION" not in by_type
    assert any("constant" in item for item in limitations)


def test_scatter_insufficient_pairs_is_controlled() -> None:
    frame = pd.DataFrame({"x": [1.0, None], "y": [2.0, 3.0]})
    plan = make_plan("RELATIONSHIP", measures=[AnalyticalExpression(kind="COLUMN", column="x"), AnalyticalExpression(kind="COLUMN", column="y")])
    result, visual = _build(frame, plan, ChartFamily.SCATTER)
    by_type, _, limitations = _facts(result, visual)
    assert by_type["PAIR_COUNT"].values["pair_count"] == 1
    assert "CORRELATION" not in by_type
    assert any("fewer than two" in item for item in limitations)


def test_histogram_dominant_bin_range_and_no_normality_claim() -> None:
    frame = pd.DataFrame({"amount": [1, 1, 2, 2, 2, 9, 10, 20]})
    plan = make_plan("DISTRIBUTION", measures=[AnalyticalExpression(kind="COLUMN", column="amount")])
    result, visual = _build(frame, plan, ChartFamily.HISTOGRAM)
    by_type, _, limitations = _facts(result, visual)
    assert "DOMINANT_BIN" in by_type and "RANGE" in by_type
    assert any("normality is not inferred" in item for item in limitations)


def test_box_and_heatmap_supported_shapes() -> None:
    frame = pd.DataFrame({"category": ["A", "B"], "amount": [1.0, 2.0]})
    plan = make_plan("GROUP_AGGREGATE", measures=[AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")], dimension="category")
    result, visual = _build(frame, plan, ChartFamily.BAR)
    box_result = result.model_copy(update={"rows": [{"category": "A", "low": 1.0, "q1": 2.0, "median": 3.0, "q3": 5.0, "high": 8.0, "outlier_count": 2}]})
    box_spec = visual.chart_spec.model_copy(update={"family": ChartFamily.BOX_PLOT})
    box_plan = visual.plan.model_copy(update={"chart_family": ChartFamily.BOX_PLOT})
    by_type, _, _ = _facts(box_result, visual.model_copy(update={"plan": box_plan, "chart_spec": box_spec}))
    assert by_type["MEDIAN"].values["median"] == 3.0
    assert by_type["IQR"].values["iqr"] == 3.0
    assert by_type["OUTLIER_COUNT"].values["count"] == 2

    heat_result = result.model_copy(update={"rows": [{"category": "A", "column": "X", "score": 9.0}, {"category": "B", "column": "Y", "score": 1.0}]})
    heat_spec = visual.chart_spec.model_copy(update={"family": ChartFamily.HEATMAP, "encodings": [
        visual.plan.x_encoding,
        FieldEncoding(channel="Y", field="column", semantic_label="Column", data_type="categorical", role="DIMENSION"),
        FieldEncoding(channel="COLOR", field="score", semantic_label="Score", data_type="numeric", role="MEASURE"),
    ]})
    heat_plan = visual.plan.model_copy(update={"chart_family": ChartFamily.HEATMAP})
    by_type, _, _ = _facts(heat_result, visual.model_copy(update={"plan": heat_plan, "chart_spec": heat_spec}))
    assert by_type["HIGH_VALUE_CELL"].values["value"] == 9.0
    assert by_type["LOW_VALUE_CELL"].values["value"] == 1.0


def test_unsupported_verified_shape_returns_controlled_xai_status() -> None:
    frame = pd.DataFrame({"category": ["A", "B"], "amount": [1.0, 2.0]})
    plan = make_plan("GROUP_AGGREGATE", measures=[AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")], dimension="category")
    result, visual = _build(frame, plan, ChartFamily.BAR)
    box_plan = visual.plan.model_copy(update={"chart_family": ChartFamily.BOX_PLOT})
    box_spec = visual.chart_spec.model_copy(update={"family": ChartFamily.BOX_PLOT})
    xai = VisualizationXAIService().build(result, box_plan, box_spec)
    assert xai.status == "UNSUPPORTED"
    assert any("box-summary" in item for item in xai.limitations)


@pytest.mark.parametrize("dataset", ["retail", "hr", "healthcare", "banking", "iot", "cybersecurity", "manufacturing", "generic"])
def test_cross_domain_names_do_not_change_xai_rules(dataset: str) -> None:
    frame = pd.DataFrame({"category": ["A", "B"], "amount": [1.0, 2.0]})
    knowledge = build_dataset_knowledge(None, dataset_id=dataset, dataframe_schema={str(c): str(t) for c, t in frame.dtypes.items()})
    plan = make_plan("GROUP_AGGREGATE", measures=[AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")], dimension="category")
    result = PandasAnalyticalExecutor().execute(plan, frame, knowledge)
    _, visual = build_requested_visualization(plan, result, knowledge, ChartFamily.BAR)
    assert visual is not None
    assert {item.fact_type for item in VisualizationXAIService().build(result, visual.plan, visual.chart_spec).facts} >= {"MAXIMUM", "MINIMUM"}
