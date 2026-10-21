from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from pipelines.pipeline3_results.grounding.schemas import (
    AnalyticalExpression,
    DatasetColumnKnowledge,
    DatasetKnowledge,
)
from pipelines.pipeline3_results.insights.results_service import build_insight_results
from pipelines.pipeline3_results.planning.compatibility import (
    adapt_plan_to_insight_query,
    adapt_plan_to_visualization_request,
)
from pipelines.pipeline3_results.planning.schemas import (
    AnalyticalPlan,
    GroupingExpression,
    PlanFilter,
    PlanProvenance,
    SortExpression,
)
from pipelines.pipeline3_results.planning.validator import validate_analytical_plan
from pipelines.pipeline3_results.visualization.results_service import build_visualization_results


PROVENANCE = PlanProvenance(
    source_intent={"analytical_goal": "SIMPLE_AGGREGATION"},
    grounding_status="RESOLVED",
)


def _knowledge() -> DatasetKnowledge:
    return DatasetKnowledge(columns=[
        DatasetColumnKnowledge(
            name="amount", physical_dtype="float64", semantic_type="currency",
            semantic_role="measurement", value_function="measurement",
            numeric_meaning="magnitude",
        ),
        DatasetColumnKnowledge(
            name="quantity", physical_dtype="int64", semantic_type="numeric_measurement",
            semantic_role="measurement", value_function="measurement",
            numeric_meaning="magnitude",
        ),
        DatasetColumnKnowledge(
            name="category", physical_dtype="object", semantic_type="categorical",
            semantic_role="category", value_function="classification",
        ),
        DatasetColumnKnowledge(
            name="customer_id", physical_dtype="object", semantic_type="identifier",
            semantic_role="primary_identifier", value_function="entity_reference",
        ),
        DatasetColumnKnowledge(
            name="event_date", physical_dtype="datetime64[ns]", semantic_type="date",
            semantic_role="timestamp", value_function="temporal",
        ),
    ])


def _plan(operation: str, measures: list[AnalyticalExpression], **changes) -> AnalyticalPlan:
    payload = {
        "plan_id": "plan_0123456789abcdef",
        "operation": operation,
        "measures": measures,
        "provenance": PROVENANCE,
    }
    payload.update(changes)
    return AnalyticalPlan(**payload)


def test_schema_forbids_arbitrary_plan_nodes() -> None:
    with pytest.raises(ValidationError):
        AnalyticalPlan(
            plan_id="plan_0123456789abcdef",
            operation="AGGREGATE",
            measures=[AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")],
            provenance=PROVENANCE,
            sql="SELECT * FROM dataset",
        )


@pytest.mark.parametrize(
    ("plan", "expected_error"),
    [
        (_plan("AGGREGATE", [AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="missing")]), "Unknown measure column"),
        (_plan("COUNT", [AnalyticalExpression(kind="COUNT_DISTINCT", operation="COUNT_DISTINCT", column="category")]), "requires an identifier"),
        (_plan("RELATIONSHIP", [AnalyticalExpression(kind="COLUMN", column="amount")]), "exactly two measures"),
        (_plan("TREND", [AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")]), "requires temporal grouping"),
        (_plan("TOP_N", [AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")], dimensions=[GroupingExpression(kind="GROUP_BY", column="category")], sort=[SortExpression(target="MEASURE", direction="DESC", measure_index=0)]), "explicit positive limit"),
        (_plan("PART_TO_WHOLE", [AnalyticalExpression(kind="AGGREGATION", operation="MEAN", column="amount")], dimensions=[GroupingExpression(kind="GROUP_BY", column="category")]), "additive or count"),
        (_plan("AGGREGATE", [AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")], filters=[PlanFilter(column="amount", operator="GREATER_THAN", value_type="STRING", value="large")]), "requires numeric values"),
    ],
)
def test_validator_rejects_structural_or_role_contradictions(plan, expected_error) -> None:
    result = validate_analytical_plan(plan, _knowledge())
    assert result.status == "INVALID"
    assert any(expected_error in error for error in result.errors)


def test_insight_adapter_preserves_distinct_count() -> None:
    plan = _plan(
        "COUNT",
        [AnalyticalExpression(kind="COUNT_DISTINCT", operation="COUNT_DISTINCT", column="customer_id")],
    )
    result = adapt_plan_to_insight_query(plan, original_text="How many customers?")

    assert result.status == "READY"
    assert result.query.count_mode == "COUNT_DISTINCT"
    assert result.query.measure_column == "customer_id"


def test_statistic_and_time_bucket_are_controlled_legacy_incompatibilities() -> None:
    statistic = _plan(
        "STATISTIC",
        [AnalyticalExpression(kind="STATISTIC", operation="STDDEV", column="amount")],
    )
    trend = _plan(
        "TREND",
        [AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")],
        temporal=[GroupingExpression(kind="TIME_BUCKET", column="event_date", time_granularity="MONTH")],
        sort=[SortExpression(target="TEMPORAL", direction="ASC", column="event_date")],
    )

    insight = adapt_plan_to_insight_query(statistic)
    visualization = adapt_plan_to_visualization_request(trend)
    assert insight.status == "NEEDS_CLARIFICATION"
    assert visualization.status == "NEEDS_CLARIFICATION"
    assert "PLAN_READY" in insight.signals
    assert "LEGACY_EXECUTION_UNSUPPORTED" in visualization.signals


def test_relationship_visualization_adapter_selects_no_chart() -> None:
    plan = _plan(
        "RELATIONSHIP",
        [
            AnalyticalExpression(kind="COLUMN", column="amount"),
            AnalyticalExpression(kind="COLUMN", column="quantity"),
        ],
    )
    result = adapt_plan_to_visualization_request(plan, raw_request="Compare amount and quantity")

    assert result.status == "READY"
    assert result.request.requested_chart_type is None
    assert result.request.intent == "RELATIONSHIP"
    assert result.request.requested_columns == ["amount", "quantity"]


def test_insight_service_can_consume_analytical_plan() -> None:
    plan = _plan(
        "AGGREGATE",
        [AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")],
    )
    response = build_insight_results(
        pd.DataFrame({"amount": [2.0, 3.0]}),
        "total amount",
        analytical_plan=plan,
    )

    assert response["status"] == "READY"
    assert response["result"]["value"] == 5.0
    assert f"analytical_plan:{plan.plan_id}" in response["signals"]


def test_visualization_service_can_consume_analytical_plan() -> None:
    plan = _plan(
        "RELATIONSHIP",
        [
            AnalyticalExpression(kind="COLUMN", column="amount"),
            AnalyticalExpression(kind="COLUMN", column="quantity"),
        ],
    )
    response = build_visualization_results(
        pd.DataFrame({"amount": [2.0, 3.0], "quantity": [1.0, 4.0]}),
        "compare amount and quantity",
        analytical_plan=plan,
    )

    assert response["status"] == "READY"
    assert response["visualization_count"] == 1
