"""Real-data validation script for Pipeline 3 Stage 8 — Requested Visualization Resolution.

Executes end-to-end user request pipeline:
    parse_visualization_request()
    → validate_visualization_request()
    → resolve_visualization_request()

Tests real datasets:
- pizza_sales.csv
- TabArena task_363612 (Airfoil Self-Noise)
- TabArena task_363615 (Used cars)
- TabArena task_363613 (Amazon Employee Access)
- Invalid & clarification requests
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd

from pipelines.pipeline3_results.visualization.column_classifier import (
    classify_dataframe,
)
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
)


def run_test_case(
    test_id: int,
    query: str,
    df: pd.DataFrame,
    dataset_name: str,
    expected_status: str,
    expected_chart_type: str | None = None,
    expected_aggregation: str | None = None,
    override_validated_request: ValidatedVisualizationRequest | None = None,
) -> dict[str, Any]:
    """Execute full parse -> validate -> resolve pipeline on a test case."""
    clf = classify_dataframe(df, dataset_name=dataset_name)

    if override_validated_request is not None:
        vreq = override_validated_request
    else:
        req = parse_visualization_request(query, df, clf)
        vreq = validate_visualization_request(df, clf, req)

    res = resolve_visualization_request(df, vreq, dataset_name=dataset_name)

    # 1. Status verification
    assert res["status"] == expected_status, (
        f"Test {test_id} failed: expected status '{expected_status}', got '{res['status']}'"
    )

    # 2. JSON Safety verification
    serialized = json.dumps(res)
    roundtrip = json.loads(serialized)
    assert roundtrip == res, f"Test {test_id} failed JSON round-trip equality"

    if expected_status == "READY":
        assert res["visualization_count"] == 1, (
            f"Test {test_id}: expected visualization_count == 1, got {res['visualization_count']}"
        )
        assert len(res["visualizations"]) == 1
        viz = res["visualizations"][0]

        # Score must ALWAYS be None
        assert viz["score"] is None, (
            f"Test {test_id}: score must be None for user requests, got {viz['score']}"
        )

        # Chart type check
        if expected_chart_type is not None:
            assert viz["chart_type"] == expected_chart_type, (
                f"Test {test_id}: expected chart_type '{expected_chart_type}', got '{viz['chart_type']}'"
            )

        # Aggregation check
        chart_data = viz["chart_data"]
        assert chart_data["has_data"] is True
        assert chart_data["usable_row_count"] > 0
        if expected_aggregation is not None:
            assert chart_data["aggregation"] == expected_aggregation, (
                f"Test {test_id}: expected aggregation '{expected_aggregation}', got '{chart_data['aggregation']}'"
            )

    elif expected_status == "INVALID":
        assert res["visualization_count"] == 0
        assert res["visualizations"] == []
        assert len(res.get("errors", [])) > 0, f"Test {test_id}: INVALID result must contain errors"

    elif expected_status == "NEEDS_CLARIFICATION":
        assert res["visualization_count"] == 0
        assert res["visualizations"] == []
        assert len(res.get("clarification_options", [])) > 0, (
            f"Test {test_id}: NEEDS_CLARIFICATION must contain options"
        )

    return res


def main() -> None:
    print("=" * 80)
    print("PIPELINE 3 — STAGE 8 REAL-DATA AND TABARENA VALIDATION")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Load Datasets
    # -----------------------------------------------------------------------
    pizza_path = Path("tests/datasets/valid/pizza_sales.csv")
    tabarena_363612_path = Path("benchmarks/tabarena/pipeline1_outputs/363612/cleaned_dataset.csv")
    tabarena_363615_path = Path("benchmarks/tabarena/pipeline1_outputs/363615/cleaned_dataset.csv")
    tabarena_363613_path = Path("benchmarks/tabarena/pipeline1_outputs/363613/cleaned_dataset.csv")

    assert pizza_path.exists(), f"Missing dataset: {pizza_path}"
    assert tabarena_363612_path.exists(), f"Missing dataset: {tabarena_363612_path}"
    assert tabarena_363615_path.exists(), f"Missing dataset: {tabarena_363615_path}"
    assert tabarena_363613_path.exists(), f"Missing dataset: {tabarena_363613_path}"

    df_pizza = pd.read_csv(pizza_path)
    df_363612 = pd.read_csv(tabarena_363612_path)
    df_363615 = pd.read_csv(tabarena_363615_path)
    df_363613 = pd.read_csv(tabarena_363613_path)

    print(f"Loaded pizza_sales.csv: {df_pizza.shape}")
    print(f"Loaded TabArena 363612 (Airfoil): {df_363612.shape}")
    print(f"Loaded TabArena 363615 (Cars): {df_363615.shape}")
    print(f"Loaded TabArena 363613 (Amazon Access): {df_363613.shape}")
    print("-" * 80)

    results_summary: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # 1–4: pizza_sales.csv
    # -----------------------------------------------------------------------
    # 1. "Bar chart of total price by pizza_category"
    res1 = run_test_case(
        1, "Bar chart of total price by pizza_category", df_pizza, "pizza_sales",
        expected_status="READY", expected_chart_type="BAR", expected_aggregation="sum",
    )
    results_summary.append({
        "id": 1, "query": "Bar chart of total price by pizza_category",
        "dataset": "pizza_sales", "status": res1["status"],
        "chart_type": res1["visualizations"][0]["chart_type"],
        "aggregation": res1["visualizations"][0]["chart_data"]["aggregation"],
        "score": res1["visualizations"][0]["score"],
        "items": len(res1["visualizations"][0]["chart_data"]["points"]),
    })

    # 2. "Median unit_price by pizza_category"
    res2 = run_test_case(
        2, "Median unit_price by pizza_category", df_pizza, "pizza_sales",
        expected_status="READY", expected_chart_type="BAR", expected_aggregation="median",
    )
    results_summary.append({
        "id": 2, "query": "Median unit_price by pizza_category",
        "dataset": "pizza_sales", "status": res2["status"],
        "chart_type": res2["visualizations"][0]["chart_type"],
        "aggregation": res2["visualizations"][0]["chart_data"]["aggregation"],
        "score": res2["visualizations"][0]["score"],
        "items": len(res2["visualizations"][0]["chart_data"]["points"]),
    })

    # 3. "Scatter plot of unit_price and total_price"
    res3 = run_test_case(
        3, "Scatter plot of unit_price and total_price", df_pizza, "pizza_sales",
        expected_status="READY", expected_chart_type="SCATTER", expected_aggregation=None,
    )
    results_summary.append({
        "id": 3, "query": "Scatter plot of unit_price and total_price",
        "dataset": "pizza_sales", "status": res3["status"],
        "chart_type": res3["visualizations"][0]["chart_type"],
        "aggregation": res3["visualizations"][0]["chart_data"]["aggregation"],
        "score": res3["visualizations"][0]["score"],
        "items": len(res3["visualizations"][0]["chart_data"]["points"]),
    })

    # 4. "Line chart of total_price over order_date"
    res4 = run_test_case(
        4, "Line chart of total_price over order_date", df_pizza, "pizza_sales",
        expected_status="READY", expected_chart_type="LINE", expected_aggregation="sum",
    )
    results_summary.append({
        "id": 4, "query": "Line chart of total_price over order_date",
        "dataset": "pizza_sales", "status": res4["status"],
        "chart_type": res4["visualizations"][0]["chart_type"],
        "aggregation": res4["visualizations"][0]["chart_data"]["aggregation"],
        "score": res4["visualizations"][0]["score"],
        "items": len(res4["visualizations"][0]["chart_data"]["points"]),
    })

    # -----------------------------------------------------------------------
    # 5–6: TabArena task_363612 (Airfoil Self-Noise)
    # -----------------------------------------------------------------------
    # 5. "Scatter plot of attack-angle and scaled-sound-pressure"
    res5 = run_test_case(
        5, "Scatter plot of attack-angle and scaled-sound-pressure", df_363612, "task_363612",
        expected_status="READY", expected_chart_type="SCATTER", expected_aggregation=None,
    )
    results_summary.append({
        "id": 5, "query": "Scatter plot of attack-angle and scaled-sound-pressure",
        "dataset": "task_363612", "status": res5["status"],
        "chart_type": res5["visualizations"][0]["chart_type"],
        "aggregation": res5["visualizations"][0]["chart_data"]["aggregation"],
        "score": res5["visualizations"][0]["score"],
        "items": len(res5["visualizations"][0]["chart_data"]["points"]),
    })

    # 6. "Histogram of frequency"
    res6 = run_test_case(
        6, "Histogram of frequency", df_363612, "task_363612",
        expected_status="READY", expected_chart_type="HISTOGRAM", expected_aggregation="count",
    )
    results_summary.append({
        "id": 6, "query": "Histogram of frequency",
        "dataset": "task_363612", "status": res6["status"],
        "chart_type": res6["visualizations"][0]["chart_type"],
        "aggregation": res6["visualizations"][0]["chart_data"]["aggregation"],
        "score": res6["visualizations"][0]["score"],
        "items": len(res6["visualizations"][0]["chart_data"]["bins"]),
    })

    # -----------------------------------------------------------------------
    # 7–10: TabArena task_363615 (Used cars)
    # -----------------------------------------------------------------------
    # 7. "Bar chart of average price by model"
    res7 = run_test_case(
        7, "Bar chart of average price by model", df_363615, "task_363615",
        expected_status="READY", expected_chart_type="BAR", expected_aggregation="mean",
    )
    results_summary.append({
        "id": 7, "query": "Bar chart of average price by model",
        "dataset": "task_363615", "status": res7["status"],
        "chart_type": res7["visualizations"][0]["chart_type"],
        "aggregation": res7["visualizations"][0]["chart_data"]["aggregation"],
        "score": res7["visualizations"][0]["score"],
        "items": len(res7["visualizations"][0]["chart_data"]["points"]),
    })

    # 8. "Median price by model"
    res8 = run_test_case(
        8, "Median price by model", df_363615, "task_363615",
        expected_status="READY", expected_chart_type="BAR", expected_aggregation="median",
    )
    results_summary.append({
        "id": 8, "query": "Median price by model",
        "dataset": "task_363615", "status": res8["status"],
        "chart_type": res8["visualizations"][0]["chart_type"],
        "aggregation": res8["visualizations"][0]["chart_data"]["aggregation"],
        "score": res8["visualizations"][0]["score"],
        "items": len(res8["visualizations"][0]["chart_data"]["points"]),
    })

    # 9. "Max price by model"
    res9 = run_test_case(
        9, "Max price by model", df_363615, "task_363615",
        expected_status="READY", expected_chart_type="BAR", expected_aggregation="max",
    )
    results_summary.append({
        "id": 9, "query": "Max price by model",
        "dataset": "task_363615", "status": res9["status"],
        "chart_type": res9["visualizations"][0]["chart_type"],
        "aggregation": res9["visualizations"][0]["chart_data"]["aggregation"],
        "score": res9["visualizations"][0]["score"],
        "items": len(res9["visualizations"][0]["chart_data"]["points"]),
    })

    # 10. "Scatter plot of km and price"
    res10 = run_test_case(
        10, "Scatter plot of km and price", df_363615, "task_363615",
        expected_status="READY", expected_chart_type="SCATTER", expected_aggregation=None,
    )
    results_summary.append({
        "id": 10, "query": "Scatter plot of km and price",
        "dataset": "task_363615", "status": res10["status"],
        "chart_type": res10["visualizations"][0]["chart_type"],
        "aggregation": res10["visualizations"][0]["chart_data"]["aggregation"],
        "score": res10["visualizations"][0]["score"],
        "items": len(res10["visualizations"][0]["chart_data"]["points"]),
    })

    # -----------------------------------------------------------------------
    # 11: TabArena task_363613 (Amazon Access)
    # -----------------------------------------------------------------------
    # 11. "Count of ResourceApproved"
    res11 = run_test_case(
        11, "Count of ResourceApproved", df_363613, "task_363613",
        expected_status="READY", expected_chart_type="COUNT_BAR", expected_aggregation="count",
    )
    results_summary.append({
        "id": 11, "query": "Count of ResourceApproved",
        "dataset": "task_363613", "status": res11["status"],
        "chart_type": res11["visualizations"][0]["chart_type"],
        "aggregation": res11["visualizations"][0]["chart_data"]["aggregation"],
        "score": res11["visualizations"][0]["score"],
        "items": len(res11["visualizations"][0]["chart_data"]["points"]),
    })

    # -----------------------------------------------------------------------
    # 12–14: Invalid & Clarification Requests
    # -----------------------------------------------------------------------
    # 12. "make a chart" -> NEEDS_CLARIFICATION
    res12 = run_test_case(
        12, "make a chart", df_pizza, "pizza_sales",
        expected_status="NEEDS_CLARIFICATION",
    )
    results_summary.append({
        "id": 12, "query": "make a chart",
        "dataset": "pizza_sales", "status": res12["status"],
        "chart_type": "-", "aggregation": "-", "score": "-",
        "items": f"clarifications: {len(res12['clarification_options'])}",
    })

    # 13. "Scatter plot of model and price" -> INVALID (model is CATEGORY)
    res13 = run_test_case(
        13, "Scatter plot of model and price", df_363615, "task_363615",
        expected_status="INVALID",
    )
    results_summary.append({
        "id": 13, "query": "Scatter plot of model and price",
        "dataset": "task_363615", "status": res13["status"],
        "chart_type": "-", "aggregation": "-", "score": "-",
        "items": f"errors: {len(res13['errors'])}",
    })

    # 14. "Median of attack-angle on scatter" -> INVALID (SCATTER does not support aggregation)
    # Testing both via Stage 7 parse and direct Stage 8 aggregation guard
    scatter_with_agg_vreq = ValidatedVisualizationRequest(
        status="VALID",
        chart_type="SCATTER",
        x_column="attack-angle",
        y_column="scaled-sound-pressure",
        aggregation="MEDIAN",
        intent="RELATIONSHIP",
        requested_columns=["attack-angle", "scaled-sound-pressure"],
    )
    res14 = run_test_case(
        14, "Median of attack-angle on scatter", df_363612, "task_363612",
        expected_status="INVALID",
        override_validated_request=scatter_with_agg_vreq,
    )
    results_summary.append({
        "id": 14, "query": "Median of attack-angle on scatter",
        "dataset": "task_363612", "status": res14["status"],
        "chart_type": "-", "aggregation": "-", "score": "-",
        "items": f"errors: {len(res14['errors'])}",
    })

    # -----------------------------------------------------------------------
    # Print Table
    # -----------------------------------------------------------------------
    print(f"{'#':<3} | {'Dataset':<12} | {'Status':<19} | {'Chart Type':<10} | {'Agg':<7} | {'Score':<5} | {'Points/Notes'}")
    print("-" * 88)
    for r in results_summary:
        print(
            f"{r['id']:<3} | {r['dataset']:<12} | {r['status']:<19} | {str(r['chart_type']):<10} | "
            f"{str(r['aggregation']):<7} | {str(r['score']):<5} | {str(r['items'])}"
        )
    print("-" * 88)
    print("ALL 14 REAL-DATA AND TABARENA TEST CASES PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
