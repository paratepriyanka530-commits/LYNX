"""End-to-end tests for the deterministic Insights foundation."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from pipelines.pipeline3_results.insights.results_service import build_insight_results
from pipelines.pipeline3_results.insights.schemas import InsightQuery


@pytest.fixture
def insight_df() -> pd.DataFrame:
    return pd.DataFrame({
        "salary": [50.0, 70.0, 60.0, 80.0, None, 90.0, 55.0, 75.0],
        "revenue": [100.0, 200.0, 50.0, 300.0, 150.0, 250.0, 125.0, 175.0],
        "order_value": [10.0, 30.0, 20.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        "department": ["A", "A", "B", "B", "C", "C", "D", "D"],
        "region": ["East", "East", "West", "West", "North", "North", "South", "South"],
        "product": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
        "score": [3.0, 5.0, 2.0, 4.0, 1.0, 6.0, 2.5, 3.5],
        "customer_id": ["C1", "C2", "C1", "C3", "C2", "C4", "C4", "C5"],
        "order_id": [f"O{i}" for i in range(8)],
        "category": ["X", "X", "Y", "Y", "Z", "Z", "X", "Y"],
        "discount": [1.0, 2.0, 1.5, 2.5, 3.0, 2.2, 1.8, 2.7],
    })


def test_average_salary_is_computed_from_actual_values(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(insight_df, "What is the average salary?")

    assert result["status"] == "READY"
    assert result["result"]["aggregation"] == "MEAN"
    assert result["result"]["value"] == pytest.approx(480.0 / 7.0)


def test_total_revenue_is_correct(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(insight_df, "What is total revenue?")

    assert result["status"] == "READY"
    assert result["result"]["aggregation"] == "SUM"
    assert result["result"]["value"] == 1350.0


def test_median_order_value_is_correct(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(insight_df, "What is the median order value?")

    assert result["status"] == "READY"
    assert result["result"]["value"] == 45.0


def test_highest_average_salary_group_is_correct(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(
        insight_df, "Which department has the highest average salary?"
    )

    assert result["status"] == "READY"
    assert result["result"]["winner"] == "C"
    assert result["result"]["value"] == 90.0
    assert "highest average salary" in result["answer"]


def test_lowest_total_sales_group_is_correct(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(
        insight_df, "Which region has the lowest total sales?"
    )

    assert result["status"] == "READY"
    # East and South tie at 300; stable group-label order breaks the result tie.
    assert result["result"]["winner"] == "East"
    assert result["result"]["value"] == 300.0


def test_top_five_products_are_descending(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(
        insight_df, "What are the top 5 products by revenue?"
    )

    assert result["status"] == "READY"
    rows = result["result"]["rows"]
    assert len(rows) == 5
    assert [row["value"] for row in rows] == [300.0, 250.0, 200.0, 175.0, 150.0]
    assert [row["group"] for row in rows] == ["P4", "P6", "P2", "P8", "P5"]


def test_bottom_three_departments_are_ascending(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(
        insight_df, "Which 3 departments have the lowest average score?"
    )

    assert result["status"] == "READY"
    rows = result["result"]["rows"]
    assert len(rows) == 3
    assert [row["value"] for row in rows] == [3.0, 3.0, 3.5]
    assert [row["group"] for row in rows] == ["B", "D", "C"]


def test_customer_count_is_distinct_for_identifier(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(insight_df, "How many customers are there?")

    assert result["status"] == "READY"
    assert result["result"]["count_mode"] == "COUNT_DISTINCT"
    assert result["result"]["measure_column"] == "customer_id"
    assert result["result"]["value"] == 5


def test_record_count_is_dataframe_row_count(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(insight_df, "How many records are there?")

    assert result["status"] == "READY"
    assert result["result"]["count_mode"] == "COUNT_ROWS"
    assert result["result"]["measure_column"] is None
    assert result["result"]["value"] == len(insight_df)


def test_count_by_category_uses_distinct_identifier_semantics(
    insight_df: pd.DataFrame,
) -> None:
    dataframe = insight_df[["order_id", "category"]].copy()
    report = {
        "semantic_profile": {"columns": [{
            "column_name": "order_id",
            "semantic_type": "identifier",
            "concept_description": "Unique order identifier",
        }]}
    }

    result = build_insight_results(
        dataframe,
        "Which category has the most orders?",
        pipeline1_report=report,
    )

    assert result["status"] == "READY"
    assert result["result"]["aggregation"] == "COUNT"
    assert result["result"]["count_mode"] == "COUNT_DISTINCT"
    assert result["result"]["winner"] == "X"
    assert result["result"]["value"] == 3


def test_missing_measure_is_invalid(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(insight_df, "What is the average bonus?")

    assert result["status"] == "INVALID"
    assert result["result"] is None
    assert result["errors"]


def test_sum_on_categorical_column_is_invalid(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(insight_df, "What is total department?")

    assert result["status"] == "INVALID"
    assert any("requires a MEASURE" in error for error in result["errors"])


def test_ambiguous_sales_needs_clarification() -> None:
    dataframe = pd.DataFrame({
        "gross_sales": [10.0, 20.0, 30.0, 40.0],
        "net_sales": [8.0, 16.0, 24.0, 32.0],
    })

    result = build_insight_results(dataframe, "What is total sales?")

    assert result["status"] == "NEEDS_CLARIFICATION"
    assert result["clarification_options"] == ["gross_sales", "net_sales"]
    assert result["result"] is None


def test_unknown_insight_request_is_controlled(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(insight_df, "tell me something")

    assert result["status"] == "NEEDS_CLARIFICATION"
    assert result["result"] is None


def test_visualization_request_is_not_executed_as_insight(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(insight_df, "show salary by department")

    assert result["status"] == "NEEDS_CLARIFICATION"
    assert "visualization request" in result["message"].lower()


def test_filter_is_not_silently_ignored(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(
        insight_df, "What is total revenue in north region?"
    )

    assert result["status"] == "NEEDS_CLARIFICATION"
    assert "filter" in result["message"].lower()
    assert result["result"] is None


def test_relationship_question_is_not_executed(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(
        insight_df, "Is discount related to revenue?"
    )

    assert result["status"] == "NEEDS_CLARIFICATION"
    assert "relationship" in result["message"].lower()
    assert result["result"] is None


def test_null_coverage_is_reported(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(insight_df, "What is the average salary?")

    evidence = result["evidence"]
    assert evidence["total_rows"] == 8
    assert evidence["usable_rows"] == 7
    assert evidence["excluded_null_rows"] == 1


def test_grouped_evidence_contains_all_computed_values(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(
        insight_df, "Which department has the highest average salary?"
    )

    grouped = result["evidence"]["grouped_values"]
    assert len(grouped) == 4
    assert {row["group"] for row in grouped} == {"A", "B", "C", "D"}
    assert all(isinstance(row["value"], (int, float)) for row in grouped)


def test_repeated_results_are_deterministic(insight_df: pd.DataFrame) -> None:
    expected = build_insight_results(
        insight_df, "What are the top 5 products by revenue?"
    )

    for _ in range(5):
        assert build_insight_results(
            insight_df, "What are the top 5 products by revenue?"
        ) == expected


def test_input_dataframe_is_unchanged(insight_df: pd.DataFrame) -> None:
    before = insight_df.copy(deep=True)

    build_insight_results(
        insight_df, "Which department has the highest average salary?"
    )

    pd.testing.assert_frame_equal(insight_df, before)


def test_response_is_json_serializable(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(insight_df, "What is total revenue?")

    encoded = json.dumps(result)
    assert json.loads(encoded) == result


def test_opaque_columns_resolve_through_pipeline1_semantics() -> None:
    dataframe = pd.DataFrame({
        "amt": [50.0, 70.0, 60.0, 80.0, 90.0, 100.0],
        "dept_cd": ["A", "A", "B", "B", "C", "C"],
    })
    report = {
        "semantic_profile": {"columns": [
            {
                "column_name": "amt",
                "semantic_type": "numeric_measurement",
                "concept_description": "Annual employee salary",
            },
            {
                "column_name": "dept_cd",
                "semantic_type": "categorical",
                "concept_description": "Employee department name",
            },
        ]}
    }

    result = build_insight_results(
        dataframe,
        "Which department has the highest average salary?",
        pipeline1_report=report,
    )

    assert result["status"] == "READY"
    assert result["result"]["measure_column"] == "amt"
    assert result["result"]["dimension_column"] == "dept_cd"
    assert result["result"]["winner"] == "C"


def test_empty_question_is_controlled(insight_df: pd.DataFrame) -> None:
    result = build_insight_results(insight_df, "   ")

    assert result["status"] == "NEEDS_CLARIFICATION"
    assert result["result"] is None


def test_insight_query_schema_is_frozen() -> None:
    query = InsightQuery(
        original_text="How many records are there?",
        query_type="SIMPLE_AGGREGATION",
        aggregation="COUNT",
        count_mode="COUNT_ROWS",
    )

    with pytest.raises(ValidationError):
        query.aggregation = "SUM"  # type: ignore[misc]
