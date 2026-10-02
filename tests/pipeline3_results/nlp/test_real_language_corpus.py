"""User-language regressions for deterministic Pipeline 3 interpretation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipelines.pipeline3_results.insights.results_service import build_insight_results
from pipelines.pipeline3_results.visualization.results_service import (
    build_visualization_results,
)


_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def pizza() -> tuple[pd.DataFrame, dict[str, object]]:
    dataframe = pd.read_csv(_ROOT / "tests" / "datasets" / "valid" / "pizza_sales.csv")
    report = json.loads((
        _ROOT
        / "tests"
        / "integration"
        / "pipeline1_architecture_b_pizza_output_v3"
        / "cleaning_report.json"
    ).read_text(encoding="utf-8"))
    return dataframe, report


@pytest.mark.parametrize(
    ("question", "winner", "dimension"),
    [
        ("Which category has the highest sales?", "Classic", "pizza_category"),
        ("Which pizza category sells the most?", "Classic", "pizza_category"),
        ("What is the least selling category?", "Veggie", "pizza_category"),
        ("Which category brought in the most money?", "Classic", "pizza_category"),
        ("Which category generated the least revenue?", "Veggie", "pizza_category"),
        ("What category has the highest total value?", "Classic", "pizza_category"),
        ("Which pizza size generated the most revenue?", "L", "pizza_size"),
    ],
)
def test_natural_sales_comparisons_resolve_from_evidence(
    pizza, question: str, winner: str, dimension: str
) -> None:
    dataframe, report = pizza
    result = build_insight_results(dataframe, question, report, "pizza_sales")

    assert result["status"] == "READY"
    assert result["result"]["measure_column"] == "total_price"
    assert result["result"]["dimension_column"] == dimension
    assert result["result"]["winner"] == winner


def test_how_many_orders_prefers_semantic_identifier(pizza) -> None:
    dataframe, report = pizza
    result = build_insight_results(
        dataframe, "How many orders are there?", report, "pizza_sales"
    )

    assert result["status"] == "READY"
    assert result["result"]["measure_column"] == "order_id"
    assert result["result"]["count_mode"] == "COUNT_DISTINCT"


def test_ordered_most_uses_grouped_row_count(pizza) -> None:
    dataframe, report = pizza
    result = build_insight_results(
        dataframe, "Which pizza is ordered the most?", report, "pizza_sales"
    )

    assert result["status"] == "READY"
    assert result["result"]["dimension_column"] == "pizza_name"
    assert result["result"]["measure_column"] is None
    assert result["result"]["count_mode"] == "COUNT_ROWS"
    assert result["result"]["winner"] == "The Classic Deluxe Pizza"


@pytest.mark.parametrize(
    "query_text",
    [
        "Show revenue by pizza category",
        "Show total sales for each pizza category",
    ],
)
def test_natural_grouped_visualizations(pizza, query_text: str) -> None:
    dataframe, report = pizza
    result = build_visualization_results(
        dataframe, query_text, report, "pizza_sales"
    )

    assert result["status"] == "READY"
    visualization = result["visualizations"][0]
    assert visualization["x_column"] == "pizza_category"
    assert visualization["y_column"] == "total_price"


def test_doughnut_request_returns_frontend_safe_pie_payload(pizza) -> None:
    dataframe, report = pizza
    result = build_visualization_results(
        dataframe,
        "Give me revenue by category as a doughnut chart",
        report,
        "pizza_sales",
    )

    assert result["status"] == "READY"
    visualization = result["visualizations"][0]
    assert visualization["chart_type"] == "PIE"
    assert visualization["presentation_variant"] == "DOUGHNUT"
    assert visualization["chart_data"]["chart_type"] == "PIE"
    assert visualization["chart_data"]["presentation_variant"] == "DOUGHNUT"
    assert visualization["chart_data"]["aggregation"] == "sum"
    assert all(point["y"] >= 0 for point in visualization["chart_data"]["points"])


def test_value_ambiguity_preserves_resolved_dimension(pizza) -> None:
    dataframe, report = pizza
    result = build_insight_results(
        dataframe, "Which category has the highest value?", report, "pizza_sales"
    )

    assert result["status"] == "NEEDS_CLARIFICATION"
    assert result["failure_code"] == "AMBIGUOUS_CONCEPT"
    assert set(result["clarification_options"]) == {
        "quantity", "unit_price", "total_price",
    }
    assert "'value' could refer to" in result["message"]


def test_sales_ambiguity_is_not_guessed() -> None:
    dataframe = pd.DataFrame({
        "category": ["A", "B", "A", "B"],
        "gross_sales": [1.0, 2.0, 3.0, 4.0],
        "net_sales": [0.8, 1.8, 2.8, 3.8],
    })
    result = build_insight_results(
        dataframe, "Which category has the highest sales?"
    )

    assert result["status"] == "NEEDS_CLARIFICATION"
    assert result["failure_code"] == "AMBIGUOUS_CONCEPT"
    assert set(result["clarification_options"]) == {"gross_sales", "net_sales"}


def test_broad_value_ambiguity_is_not_guessed() -> None:
    dataframe = pd.DataFrame({
        "category": ["A", "B", "A", "B"],
        "unit_price": [1.0, 2.0, 3.0, 4.0],
        "total_price": [2.0, 4.0, 6.0, 8.0],
        "quantity": [1.0, 2.0, 1.0, 2.0],
    })
    result = build_insight_results(dataframe, "Which category has the highest value?")

    assert result["status"] == "NEEDS_CLARIFICATION"
    assert set(result["clarification_options"]) == {
        "unit_price", "total_price", "quantity",
    }


def test_failure_taxonomy_distinguishes_unknown_and_unsupported_chart(pizza) -> None:
    dataframe, report = pizza
    unknown = build_insight_results(
        dataframe, "What is total profit?", report, "pizza_sales"
    )
    unsupported_chart = build_visualization_results(
        dataframe, "Show revenue by category in a radar chart", report, "pizza_sales"
    )

    assert unknown["status"] == "INVALID"
    assert unknown["failure_code"] == "UNRESOLVED_CONCEPT"
    assert "profit" in unknown["message"]
    assert unsupported_chart["status"] == "INVALID"
    assert unsupported_chart["failure_code"] == "UNSUPPORTED_CHART"
    assert "radar chart" in unsupported_chart["message"]


def test_failure_taxonomy_distinguishes_operation_and_analysis() -> None:
    dataframe = pd.DataFrame({
        "category": ["A", "B", "A", "B"],
        "revenue": [10.0, 20.0, 30.0, 40.0],
        "quantity": [1.0, 2.0, 3.0, 4.0],
    })
    unsupported_operation = build_insight_results(
        dataframe, "What is total revenue where category is A?"
    )
    unsupported_analysis = build_insight_results(
        dataframe, "Is revenue related to quantity?"
    )

    assert unsupported_operation["status"] == "NEEDS_CLARIFICATION"
    assert unsupported_operation["failure_code"] == "UNSUPPORTED_OPERATION"
    assert unsupported_analysis["status"] == "NEEDS_CLARIFICATION"
    assert unsupported_analysis["failure_code"] == "UNSUPPORTED_ANALYSIS"


def test_pie_rejects_negative_measure_values() -> None:
    dataframe = pd.DataFrame({
        "category": ["A", "B", "C"],
        "revenue": [10.0, -2.0, 5.0],
    })
    result = build_visualization_results(
        dataframe, "total revenue by category as a pie chart"
    )

    assert result["status"] == "INVALID"
    assert result["failure_code"] == "INVALID_COMBINATION"
    assert any("negative values" in error for error in result["errors"])


def test_pie_rejects_high_cardinality_dimension() -> None:
    dataframe = pd.DataFrame({
        "category": [f"C{index}" for index in range(13) for _ in range(2)],
        "revenue": [float(index + 1) for index in range(13) for _ in range(2)],
    })
    result = build_visualization_results(
        dataframe, "total revenue by category as a pie chart"
    )

    assert result["status"] == "INVALID"
    assert result["failure_code"] == "INVALID_COMBINATION"
    assert any("unsuitable" in error for error in result["errors"])
