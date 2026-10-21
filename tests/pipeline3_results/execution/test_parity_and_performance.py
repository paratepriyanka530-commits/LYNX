from __future__ import annotations

import math
from time import perf_counter

import pandas as pd
import pytest

from pipelines.pipeline3_results.execution.pandas_executor import PandasAnalyticalExecutor
from pipelines.pipeline3_results.grounding.schemas import AnalyticalExpression, DatasetColumnKnowledge, DatasetKnowledge
from pipelines.pipeline3_results.insights.analysis_engine import execute_insight_query
from pipelines.pipeline3_results.insights.schemas import InsightQuery
from pipelines.pipeline3_results.insights.results_service import build_insight_results
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan, GroupingExpression, PlanProvenance, SortExpression
from pipelines.pipeline3_results.visualization.results_service import build_visualization_results


KNOWLEDGE = DatasetKnowledge(columns=[
    DatasetColumnKnowledge(name="value", physical_dtype="float64", semantic_type="numeric_measurement", semantic_role="measurement", value_function="measurement", numeric_meaning="magnitude"),
    DatasetColumnKnowledge(name="group", physical_dtype="object", semantic_type="categorical", semantic_role="category", value_function="classification"),
])
PROVENANCE = PlanProvenance(grounding_status="RESOLVED")


def _plan(operation: str, expression: AnalyticalExpression, **updates) -> AnalyticalPlan:
    payload = {"plan_id":"plan_abcdef0123456789", "operation":operation, "measures":[expression], "provenance":PROVENANCE}
    payload.update(updates)
    return AnalyticalPlan(**payload)


@pytest.mark.parametrize("operation", ["SUM", "MEAN", "MEDIAN", "MIN", "MAX"])
def test_scalar_aggregation_matches_legacy_insights(operation: str) -> None:
    dataframe = pd.DataFrame({"value":[1.0, 2.0, None, 5.0], "group":["a","a","b","b"]})
    plan = _plan("AGGREGATE", AnalyticalExpression(kind="AGGREGATION", operation=operation, column="value"))
    shared = PandasAnalyticalExecutor().execute(plan, dataframe, KNOWLEDGE)
    legacy = execute_insight_query(dataframe, InsightQuery(original_text="", query_type="SIMPLE_AGGREGATION", measure_column="value", aggregation=operation))
    assert math.isclose(float(shared.scalar), float(legacy.value), rel_tol=1e-12)


@pytest.mark.parametrize(
    ("kind", "column", "count_mode"),
    [("COUNT_ROWS", None, "COUNT_ROWS"), ("COUNT_NON_NULL", "value", "COUNT_NON_NULL")],
)
def test_count_matches_legacy_insights(kind, column, count_mode) -> None:
    dataframe = pd.DataFrame({"value":[1.0, None, 3.0], "group":["a","a","b"]})
    plan = _plan("COUNT", AnalyticalExpression(kind=kind, column=column))
    shared = PandasAnalyticalExecutor().execute(plan, dataframe, KNOWLEDGE)
    legacy = execute_insight_query(dataframe, InsightQuery(original_text="", query_type="SIMPLE_AGGREGATION", measure_column=column, aggregation="COUNT", count_mode=count_mode))
    assert shared.scalar == legacy.value


def test_grouped_sum_matches_legacy_insights_by_group() -> None:
    dataframe = pd.DataFrame({"value":[1.0, 2.0, 4.0], "group":["a","a","b"]})
    expression = AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="value")
    plan = _plan("GROUP_AGGREGATE", expression, dimensions=[GroupingExpression(kind="GROUP_BY", column="group")])
    shared = PandasAnalyticalExecutor().execute(plan, dataframe, KNOWLEDGE)
    legacy = execute_insight_query(dataframe, InsightQuery(original_text="", query_type="GROUPED_AGGREGATION", measure_column="value", dimension_column="group", aggregation="SUM"))
    assert {row["group"]:row["value"] for row in shared.rows} == {row.group:row.value for row in legacy.rows}


def test_highest_rank_matches_legacy_when_there_is_no_tie() -> None:
    dataframe = pd.DataFrame({"value":[1.0, 2.0, 4.0], "group":["a","a","b"]})
    expression = AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="value")
    plan = _plan("RANK", expression, dimensions=[GroupingExpression(kind="GROUP_BY", column="group")], sort=[SortExpression(target="MEASURE", direction="DESC", measure_index=0)], comparison="HIGHEST")
    shared = PandasAnalyticalExecutor().execute(plan, dataframe, KNOWLEDGE)
    legacy = execute_insight_query(dataframe, InsightQuery(original_text="", query_type="GROUPED_COMPARISON", measure_column="value", dimension_column="group", aggregation="SUM", comparison="HIGHEST"))
    assert shared.rows[0]["group"] == legacy.winner
    assert shared.rows[0]["value"] == legacy.value


def test_insights_and_visualization_consume_the_same_grouped_truth() -> None:
    dataframe = pd.DataFrame({"value":[1.0, 2.0, 4.0], "group":["a","a","b"]})
    plan = _plan(
        "GROUP_AGGREGATE",
        AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="value"),
        dimensions=[GroupingExpression(kind="GROUP_BY", column="group")],
    )
    insight = build_insight_results(dataframe, "value by group", analytical_plan=plan)
    visualization = build_visualization_results(dataframe, "value by group", analytical_plan=plan)

    insight_values = {row["group"]: row["value"] for row in insight["result"]["rows"]}
    chart_values = {
        point["x"]: point["y"]
        for point in visualization["visualizations"][0]["chart_data"]["points"]
    }
    assert insight_values == chart_values == {"a": 3.0, "b": 4.0}
    assert "visualization_compatibility:no_recomputation" in visualization["visualizations"][0]["signals"]


@pytest.mark.parametrize(
    ("values", "error_code"),
    [([5.0, -1.0], "NEGATIVE_CONTRIBUTION"), ([0.0, 0.0], "ZERO_DENOMINATOR")],
)
def test_part_to_whole_rejects_invalid_contribution_semantics(values, error_code) -> None:
    dataframe = pd.DataFrame({"value":values, "group":["a","b"]})
    plan = _plan(
        "PART_TO_WHOLE",
        AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="value"),
        dimensions=[GroupingExpression(kind="GROUP_BY", column="group")],
    )
    result = PandasAnalyticalExecutor().execute(plan, dataframe, KNOWLEDGE)
    assert result.status == "INVALID"
    assert result.errors[0].code == error_code


@pytest.mark.parametrize("row_count", [100, 50_000])
def test_bounded_synthetic_execution_performance(row_count: int) -> None:
    dataframe = pd.DataFrame({"value":[float(index % 100) for index in range(row_count)], "group":[f"g{index % 25}" for index in range(row_count)]})
    plan = _plan("GROUP_AGGREGATE", AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="value"), dimensions=[GroupingExpression(kind="GROUP_BY", column="group")])
    started = perf_counter()
    result = PandasAnalyticalExecutor().execute(plan, dataframe, KNOWLEDGE)
    elapsed = perf_counter() - started
    assert result.status == "READY"
    assert elapsed < 10.0
