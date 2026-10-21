from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from pipelines.pipeline3_results.execution.pandas_executor import PandasAnalyticalExecutor
from pipelines.pipeline3_results.grounding.dataset_knowledge import build_dataset_knowledge
from pipelines.pipeline3_results.grounding.schemas import AnalyticalExpression
from pipelines.pipeline3_results.insights.landing.opportunities import make_plan
from pipelines.pipeline3_results.visualization.intelligence.schemas import (
    ChartFamily, RendererNeutralChartSpec, VisualizationPlan, VisualizationPurpose,
)
from pipelines.pipeline3_results.visualization.intelligence.service import (
    VisualizationIntelligenceService, build_requested_visualization,
)


def _frame(categories: int = 4) -> pd.DataFrame:
    rows = max(40, categories)
    return pd.DataFrame({
        "event_date": pd.date_range("2025-01-01", periods=rows),
        "category": [f"C{i % categories}" for i in range(rows)],
        "amount": [float(i + 1) for i in range(rows)],
        "quantity": [float((i % 7) + 1) for i in range(rows)],
    })


def _execute(df: pd.DataFrame, plan):
    knowledge = build_dataset_knowledge(None, dataset_id="test", dataframe_schema={str(c): str(t) for c, t in df.dtypes.items()})
    return knowledge, PandasAnalyticalExecutor().execute(plan, df, knowledge)


def test_contracts_are_frozen_and_renderer_neutral() -> None:
    assert len(ChartFamily) == 18
    assert {"TREND", "FLOW", "KPI_SUPPORT"} <= {item.value for item in VisualizationPurpose}
    forbidden = {"xAxis", "yAxis", "dom", "css", "component"}
    assert not forbidden.intersection(RendererNeutralChartSpec.model_fields)
    assert not forbidden.intersection(VisualizationPlan.model_fields)


def test_automatic_dashboard_is_diverse_bounded_and_verified() -> None:
    dashboard = VisualizationIntelligenceService().build_dashboard("retail", _frame())
    assert 3 <= len(dashboard.visualizations) <= 5
    purposes = [item.plan.purpose for item in dashboard.visualizations]
    assert len(purposes) == len(set(purposes))
    assert {"TREND", "RANKING", "DISTRIBUTION", "RELATIONSHIP"} <= {item.value for item in purposes}
    assert all(item.chart_spec.data_ref.source_result_id == item.verified_result_ref for item in dashboard.visualizations)
    assert all(item.score_components for item in dashboard.visualizations)


def test_one_numeric_field_has_no_relationship_and_no_time_has_no_trend() -> None:
    df = pd.DataFrame({"category": ["A", "B"] * 20, "amount": range(40)})
    dashboard = VisualizationIntelligenceService().build_dashboard("generic", df)
    purposes = {item.plan.purpose.value for item in dashboard.visualizations}
    assert "RELATIONSHIP" not in purposes
    assert "TREND" not in purposes


def test_explicit_doughnut_is_preserved_when_suitable() -> None:
    df = _frame(4)
    plan = make_plan("PART_TO_WHOLE", measures=[AnalyticalExpression(kind="COUNT_ROWS")], dimension="category")
    knowledge, result = _execute(df, plan)
    assessment, visual = build_requested_visualization(plan, result, knowledge, ChartFamily.DOUGHNUT)
    assert assessment.status == "SUITABLE"
    assert visual is not None
    assert visual.chart_spec.family == ChartFamily.DOUGHNUT


def test_explicit_pie_warns_then_stops_at_excessive_cardinality() -> None:
    medium = _frame(12)
    plan = make_plan("PART_TO_WHOLE", measures=[AnalyticalExpression(kind="COUNT_ROWS")], dimension="category")
    knowledge, result = _execute(medium, plan)
    assessment, visual = build_requested_visualization(plan, result, knowledge, ChartFamily.PIE)
    assert assessment.status == "SUITABLE_WITH_WARNING"
    assert visual is not None and visual.plan.warnings

    large = _frame(35)
    knowledge, result = _execute(large, plan)
    assessment, visual = build_requested_visualization(plan, result, knowledge, ChartFamily.PIE)
    assert assessment.status == "NOT_RECOMMENDED"
    assert visual is None


def test_invalid_explicit_scatter_and_line_are_controlled() -> None:
    df = _frame()
    comparison = make_plan("GROUP_AGGREGATE", measures=[AnalyticalExpression(kind="AGGREGATION", operation="MEAN", column="amount")], dimension="category")
    knowledge, result = _execute(df, comparison)
    for family, code in ((ChartFamily.SCATTER, "SCATTER_REQUIRES_NUMERIC_RELATIONSHIP"), (ChartFamily.LINE, "LINE_REQUIRES_ORDERED_TEMPORAL_RESULT")):
        assessment, visual = build_requested_visualization(comparison, result, knowledge, family)
        assert assessment.status == "INVALID"
        assert assessment.reason_codes == [code]
        assert visual is None


@pytest.mark.parametrize(
    ("plan", "family"),
    [
        (make_plan("TREND", measures=[AnalyticalExpression(kind="AGGREGATION", operation="MEAN", column="amount")], temporal="event_date"), ChartFamily.LINE),
        (make_plan("DISTRIBUTION", measures=[AnalyticalExpression(kind="COLUMN", column="amount")]), ChartFamily.HISTOGRAM),
        (make_plan("RELATIONSHIP", measures=[AnalyticalExpression(kind="COLUMN", column="amount"), AnalyticalExpression(kind="COLUMN", column="quantity")]), ChartFamily.SCATTER),
        (make_plan("RANK", measures=[AnalyticalExpression(kind="COUNT_ROWS")], dimension="category", comparison="HIGHEST"), ChartFamily.BAR),
    ],
)
def test_requested_supported_families_converge(plan, family: ChartFamily) -> None:
    knowledge, result = _execute(_frame(), plan)
    assessment, visual = build_requested_visualization(plan, result, knowledge, family)
    assert assessment.status == "SUITABLE"
    assert visual is not None and visual.chart_spec.family == family


def test_unspecified_requested_chart_selects_family_after_execution() -> None:
    plan = make_plan("TREND", measures=[AnalyticalExpression(kind="AGGREGATION", operation="MEAN", column="amount")], temporal="event_date")
    knowledge, result = _execute(_frame(), plan)
    _, visual = build_requested_visualization(plan, result, knowledge)
    assert visual is not None and visual.chart_spec.family == ChartFamily.LINE


def test_additive_grouped_result_can_honor_explicit_pie() -> None:
    plan = make_plan("GROUP_AGGREGATE", measures=[AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")], dimension="category")
    knowledge, result = _execute(_frame(), plan)
    assessment, visual = build_requested_visualization(plan, result, knowledge, ChartFamily.PIE)
    assert assessment.status == "SUITABLE"
    assert visual is not None and visual.plan.purpose == VisualizationPurpose.PART_TO_WHOLE


@pytest.mark.parametrize("dataset_id", ["retail", "hr", "healthcare", "banking", "iot", "cybersecurity", "manufacturing", "survey"])
def test_cross_domain_names_do_not_change_rules(dataset_id: str) -> None:
    dashboard = VisualizationIntelligenceService().build_dashboard(dataset_id, _frame())
    assert [item.plan.purpose for item in dashboard.visualizations] == [
        item.plan.purpose for item in VisualizationIntelligenceService().build_dashboard("generic", _frame()).visualizations
    ]


def test_chart_data_bounding_is_explicit() -> None:
    df = pd.DataFrame({"left": range(700), "right": range(700, 1400)})
    plan = make_plan("RELATIONSHIP", measures=[AnalyticalExpression(kind="COLUMN", column="left"), AnalyticalExpression(kind="COLUMN", column="right")])
    knowledge, result = _execute(df, plan)
    _, visual = build_requested_visualization(plan, result, knowledge)
    assert visual is not None
    assert len(visual.chart_spec.data_ref.rows) == 500
    assert visual.chart_spec.data_ref.truncated is True
    assert visual.chart_spec.data_ref.display_limit == 500


def test_kpis_are_verified_and_non_redundant() -> None:
    dashboard = VisualizationIntelligenceService().build_dashboard("generic", _frame())
    expressions = [item.expression["kind"] for item in dashboard.kpis]
    assert len(expressions) == len(set(expressions))
    assert all(item.verified_result_id.startswith("result_") for item in dashboard.kpis)
    assert "COUNT_ROWS" in expressions
