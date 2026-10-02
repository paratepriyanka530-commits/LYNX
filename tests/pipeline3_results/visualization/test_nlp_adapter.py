"""Focused Phase 7 tests for shared-NLP visualization compatibility."""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd

from pipelines.pipeline3_results.nlp.orchestrator import interpret_query
from pipelines.pipeline3_results.visualization.column_classifier import (
    classify_dataframe,
)
from pipelines.pipeline3_results.visualization.nlp_adapter import (
    adapt_query_interpretation,
)
from pipelines.pipeline3_results.visualization.results_service import (
    build_visualization_results,
)


def _sales_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "salary": [40_000, 52_000, 61_000, 47_000, 70_000, 58_000, 63_000, 49_000],
        "department": ["A", "B", "A", "B", "C", "C", "A", "B"],
        "sales": [100.0, 125.0, 110.0, 140.0, 175.0, 160.0, 155.0, 145.0],
        "region": ["North", "South", "North", "South", "West", "West", "North", "South"],
        "discount": [1.0, 2.0, 1.5, 2.5, 3.0, 2.2, 1.8, 2.7],
        "revenue": [90.0, 115.0, 102.0, 128.0, 158.0, 149.0, 143.0, 131.0],
        "product": ["P1", "P2", "P1", "P3", "P2", "P3", "P1", "P2"],
    })


def _adapt(
    text: str,
    dataframe: pd.DataFrame,
    report: object = None,
):
    classification = classify_dataframe(dataframe, pipeline1_report=report)
    interpretation = interpret_query(
        text,
        dataframe,
        pipeline1_report=report,
        classification=classification,
        context="visualization",
    )
    return adapt_query_interpretation(interpretation, classification)


def test_average_salary_maps_to_mean_bar_request() -> None:
    adapted = _adapt(
        "show average salary by department", _sales_frame()
    )

    assert adapted.status == "READY"
    assert adapted.request is not None
    assert adapted.request.x_column == "department"
    assert adapted.request.y_column == "salary"
    assert adapted.request.aggregation == "MEAN"
    assert adapted.request.requested_chart_type is None
    assert adapted.request.intent == "COMPARISON"


def test_ambiguous_sales_returns_choices_without_request() -> None:
    dataframe = pd.DataFrame({
        "gross_sales": [10.0, 20.0, 30.0, 40.0],
        "net_sales": [8.0, 16.0, 24.0, 32.0],
        "region": ["N", "S", "N", "S"],
    })

    adapted = _adapt("show sales by region", dataframe)

    assert adapted.status == "NEEDS_CLARIFICATION"
    assert adapted.request is None
    assert adapted.ambiguous_terms == ["sales"]
    assert adapted.clarification_options == ["gross_sales", "net_sales"]


def test_opaque_semantic_columns_map_to_visualization_request() -> None:
    dataframe = pd.DataFrame({
        "amt": [40_000, 52_000, 61_000, 47_000, 70_000, 58_000],
        "dept_cd": ["A", "B", "A", "B", "C", "C"],
    })
    report = {
        "semantic_profile": {"columns": [
            {
                "column_name": "amt",
                "concept_description": "Annual employee salary",
                "semantic_type": "numeric_measurement",
            },
            {
                "column_name": "dept_cd",
                "concept_description": "Employee department name",
                "semantic_type": "categorical",
            },
        ]}
    }

    adapted = _adapt("show salary by department", dataframe, report)

    assert adapted.status == "READY"
    assert adapted.request is not None
    assert adapted.request.x_column == "dept_cd"
    assert adapted.request.y_column == "amt"
    assert "resolver:semantic_description_match" in adapted.signals


def test_explicit_bar_chart_is_preserved() -> None:
    adapted = _adapt("bar chart of sales by region", _sales_frame())

    assert adapted.status == "READY"
    assert adapted.request is not None
    assert adapted.request.requested_chart_type == "BAR"
    assert adapted.request.is_explicit_chart_type is True


def test_relationship_maps_two_fields_for_existing_validator_inference() -> None:
    adapted = _adapt(
        "show relationship between discount and revenue", _sales_frame()
    )

    assert adapted.status == "READY"
    assert adapted.request is not None
    assert adapted.request.x_column == "discount"
    assert adapted.request.y_column == "revenue"
    assert adapted.request.requested_chart_type is None
    assert adapted.request.intent == "RELATIONSHIP"


def test_single_resolved_time_column_maps_to_trend_axes() -> None:
    dataframe = pd.DataFrame({
        "sales": [100.0, 125.0, 110.0, 140.0, 175.0, 160.0],
        "order_date": pd.date_range("2024-01-01", periods=6, freq="D"),
    })

    adapted = _adapt("show sales over time", dataframe)

    assert adapted.status == "READY"
    assert adapted.request is not None
    assert adapted.request.x_column == "order_date"
    assert adapted.request.y_column == "sales"
    assert adapted.request.intent == "TREND"


def test_insight_intent_is_not_executed_as_visualization() -> None:
    adapted = _adapt(
        "which region has the highest sales?", _sales_frame()
    )

    assert adapted.status == "NEEDS_CLARIFICATION"
    assert adapted.request is None
    assert "adapter:insight_intent_not_visualized" in adapted.signals


def test_unknown_request_returns_controlled_clarification() -> None:
    adapted = _adapt("tell me something", _sales_frame())

    assert adapted.status == "NEEDS_CLARIFICATION"
    assert adapted.request is None
    assert adapted.message


def test_invalid_concepts_return_invalid_without_request() -> None:
    adapted = _adapt("show rainfall by city", _sales_frame())

    assert adapted.status == "INVALID"
    assert adapted.request is None
    assert adapted.errors


def test_top_n_is_not_silently_discarded() -> None:
    adapted = _adapt("show top 5 products by revenue", _sales_frame())

    assert adapted.status == "NEEDS_CLARIFICATION"
    assert adapted.request is None
    assert "adapter:unsupported_comparison:TOP_N" in adapted.signals


def test_filter_is_not_silently_discarded() -> None:
    adapted = _adapt("sales in north region", _sales_frame())

    assert adapted.status == "NEEDS_CLARIFICATION"
    assert adapted.request is None
    assert "adapter:unsupported_filters" in adapted.signals


def test_explicit_sum_aggregation_is_preserved() -> None:
    adapted = _adapt("show total sales by region", _sales_frame())

    assert adapted.status == "READY"
    assert adapted.request is not None
    assert adapted.request.aggregation == "SUM"


def test_automatic_mode_does_not_invoke_shared_requested_nlp() -> None:
    with patch(
        "pipelines.pipeline3_results.nlp.orchestrator.interpret_query",
        side_effect=AssertionError("requested NLP must not run in automatic mode"),
    ):
        result = build_visualization_results(_sales_frame(), request_text=None)

    assert result["mode"] == "automatic"
    assert result["status"] in {"READY", "NO_MEANINGFUL_VISUALIZATION"}


def test_ambiguous_service_path_does_not_call_validator_or_resolver() -> None:
    dataframe = pd.DataFrame({
        "gross_sales": [10.0, 20.0, 30.0, 40.0],
        "net_sales": [8.0, 16.0, 24.0, 32.0],
        "region": ["N", "S", "N", "S"],
    })
    with (
        patch(
            "pipelines.pipeline3_results.visualization.results_service.validate_visualization_request",
            side_effect=AssertionError("validator must not run"),
        ),
        patch(
            "pipelines.pipeline3_results.visualization.results_service.resolve_visualization_request",
            side_effect=AssertionError("resolver must not run"),
        ),
    ):
        result = build_visualization_results(
            dataframe, request_text="show sales by region"
        )

    assert result["status"] == "NEEDS_CLARIFICATION"
    assert result["ambiguous_terms"] == ["sales"]
    assert result["clarification_options"] == ["gross_sales", "net_sales"]
    assert result["visualizations"] == []


def test_pipeline1_semantics_affect_real_requested_service_path() -> None:
    dataframe = pd.DataFrame({
        "amt": [40_000, 52_000, 61_000, 47_000, 70_000, 58_000],
        "dept_cd": ["A", "B", "A", "B", "C", "C"],
    })
    report = {
        "semantic_profile": {"columns": [
            {"column_name": "amt", "concept_description": "Annual employee salary"},
            {"column_name": "dept_cd", "concept_description": "Employee department"},
        ]}
    }

    result = build_visualization_results(
        dataframe,
        request_text="show salary by department",
        pipeline1_report=report,
    )

    assert result["status"] == "READY"
    assert result["visualizations"][0]["x_column"] == "dept_cd"
    assert result["visualizations"][0]["y_column"] == "amt"
    assert "resolver:semantic_description_match" in result["signals"]


def test_ambiguity_choice_order_is_deterministic() -> None:
    dataframe = pd.DataFrame({
        "net_sales": [8.0, 16.0, 24.0, 32.0],
        "gross_sales": [10.0, 20.0, 30.0, 40.0],
        "region": ["N", "S", "N", "S"],
    })

    outputs = [
        _adapt("show sales by region", dataframe).clarification_options
        for _ in range(5)
    ]

    assert all(output == ["net_sales", "gross_sales"] for output in outputs)


def test_requested_path_does_not_mutate_dataframe() -> None:
    dataframe = _sales_frame()
    before = dataframe.copy(deep=True)

    build_visualization_results(
        dataframe, request_text="show average salary by department"
    )

    pd.testing.assert_frame_equal(dataframe, before)


def test_adapter_and_service_outputs_are_json_serializable() -> None:
    dataframe = _sales_frame()
    adapted = _adapt("show average salary by department", dataframe)
    service_result = build_visualization_results(
        dataframe, request_text="show average salary by department"
    )

    json.dumps(adapted.model_dump(mode="json"))
    encoded = json.dumps(service_result)
    assert json.loads(encoded) == service_result
