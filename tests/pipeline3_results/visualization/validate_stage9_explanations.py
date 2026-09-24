"""Real-data validation script for Pipeline 3 Stage 9 — Visualization Explanation & XAI Evidence.

Validates Stage 9 explanations using real datasets:
1. pizza_sales.csv (automatic Stage 6 payload)
2. dataset_01_retail_transactions_raw.csv (automatic Stage 6 payload)
3. dataset_07_iot_sensor_raw.csv (automatic Stage 6 payload)
4. dataset_09_survey_data_raw.csv (automatic Stage 6 payload)
5. Stage 8 requested: "Median unit_price by pizza_category"
6. Stage 8 requested: "Line chart of total_price over order_date"
7. Invalid and clarification payloads
"""

from __future__ import annotations

import json
import re
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
from pipelines.pipeline3_results.visualization.explanation import (
    FORBIDDEN_WORDS,
    build_visualization_explanation,
)
from pipelines.pipeline3_results.visualization.integration import (
    build_visualization_payload,
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


def verify_explanation(
    exp_res: dict[str, Any],
    source_viz: dict[str, Any],
    expected_status: str = "READY",
    expected_agg_attribution: str | None = None,
) -> None:
    """Run all integrity checks on an explanation result."""
    # 1. Structure check
    assert "chart_type" in exp_res
    assert "x_column" in exp_res
    assert "y_column" in exp_res
    assert "explanation" in exp_res
    assert "signals" in exp_res

    exp = exp_res["explanation"]
    assert "why_this_chart" in exp
    assert "what_it_shows" in exp
    assert "aggregation" in exp
    assert "data_coverage" in exp
    assert "evidence" in exp

    # 2. Field preservation
    expected_chart_type = source_viz.get("chart_type") or "UNKNOWN"
    expected_x = source_viz.get("x_column") or "UNKNOWN"
    expected_y = source_viz.get("y_column")

    assert exp_res["chart_type"] == expected_chart_type
    assert exp_res["x_column"] == expected_x
    assert exp_res["y_column"] == expected_y

    # 3. JSON round-trip
    dumped = json.dumps(exp_res)
    loaded = json.loads(dumped)
    assert loaded == exp_res, "Failed JSON serialization round-trip equality"

    # 4. Forbidden language scan
    all_text = " ".join([
        exp["why_this_chart"],
        exp["what_it_shows"],
        exp["aggregation"],
        exp["data_coverage"],
        " ".join(exp["evidence"]),
        " ".join(exp_res["signals"]),
    ])
    words = re.findall(r"\b[a-zA-Z]+\b", all_text.lower())
    for w in words:
        assert w not in FORBIDDEN_WORDS, f"Forbidden word '{w}' found in: {all_text}"

    # 5. Status-specific checks
    if expected_status == "READY":
        assert len(exp["evidence"]) > 0, "READY visualizations must have non-empty evidence"
        if expected_agg_attribution is not None:
            assert expected_agg_attribution in exp["aggregation"], (
                f"Expected attribution '{expected_agg_attribution}', got '{exp['aggregation']}'"
            )
    else:
        assert exp["evidence"] == []
        assert exp["what_it_shows"] == "No displayable chart data available."


def main() -> None:
    print("=" * 80)
    print("PIPELINE 3 — STAGE 9 VISUALIZATION EXPLANATION VALIDATION")
    print("=" * 80)

    datasets_dir = Path("tests/datasets/valid")
    pizza_path = datasets_dir / "pizza_sales.csv"
    retail_path = datasets_dir / "dataset_01_retail_transactions_raw.csv"
    iot_path = datasets_dir / "dataset_07_iot_sensor_raw.csv"
    survey_path = datasets_dir / "dataset_09_survey_data_raw.csv"

    for p in [pizza_path, retail_path, iot_path, survey_path]:
        assert p.exists(), f"Missing dataset: {p}"

    results_table: list[dict[str, Any]] = []
    case_num = 1

    # -----------------------------------------------------------------------
    # Part 1: Automatic Stage 6 Payloads (Semantic/default aggregation)
    # -----------------------------------------------------------------------
    auto_datasets = [
        ("pizza_sales.csv", pizza_path),
        ("dataset_01_retail.csv", retail_path),
        ("dataset_07_iot.csv", iot_path),
        ("dataset_09_survey.csv", survey_path),
    ]

    for ds_name, ds_path in auto_datasets:
        df = pd.read_csv(ds_path)
        payload = build_visualization_payload(df, dataset_name=ds_name, max_visualizations=3)
        assert payload["status"] == "READY"
        assert len(payload["visualizations"]) > 0

        for viz in payload["visualizations"]:
            exp_res = build_visualization_explanation(viz, dataset_signals=payload.get("signals"))
            chart_type = viz["chart_type"]
            expected_attr = "None / no explicit" if chart_type == "SCATTER" else "Semantic/default"
            verify_explanation(exp_res, viz, expected_status="READY", expected_agg_attribution=expected_attr)

            results_table.append({
                "case": case_num,
                "dataset": ds_name,
                "mode": "Stage 6 (Auto)",
                "chart_type": exp_res["chart_type"],
                "columns": f"{exp_res['x_column']} × {exp_res['y_column'] or '-'}",
                "aggregation": exp_res["explanation"]["aggregation"],
                "evidence_count": len(exp_res["explanation"]["evidence"]),
            })
            case_num += 1

    # -----------------------------------------------------------------------
    # Part 2: Stage 8 Requested Visualizations (User-requested aggregation)
    # -----------------------------------------------------------------------
    df_pizza = pd.read_csv(pizza_path)
    clf_pizza = classify_dataframe(df_pizza)

    # Request A: "Median unit_price by pizza_category"
    req_a = parse_visualization_request("Median unit_price by pizza_category", df_pizza, clf_pizza)
    vreq_a = validate_visualization_request(df_pizza, clf_pizza, req_a)
    assert vreq_a.status == "VALID"
    res_a = resolve_visualization_request(df_pizza, vreq_a, dataset_name="pizza_sales")
    assert res_a["status"] == "READY"

    viz_a = res_a["visualizations"][0]
    exp_a = build_visualization_explanation(viz_a, dataset_signals=res_a.get("signals"))
    verify_explanation(exp_a, viz_a, expected_status="READY", expected_agg_attribution="User-requested MEDIAN")

    results_table.append({
        "case": case_num,
        "dataset": "pizza_sales.csv",
        "mode": "Stage 8 (User Req)",
        "chart_type": exp_a["chart_type"],
        "columns": f"{exp_a['x_column']} × {exp_a['y_column'] or '-'}",
        "aggregation": exp_a["explanation"]["aggregation"],
        "evidence_count": len(exp_a["explanation"]["evidence"]),
    })
    case_num += 1

    # Request B: "Line chart of total_price over order_date"
    req_b = parse_visualization_request("Line chart of total_price over order_date", df_pizza, clf_pizza)
    vreq_b = validate_visualization_request(df_pizza, clf_pizza, req_b)
    assert vreq_b.status == "VALID"
    res_b = resolve_visualization_request(df_pizza, vreq_b, dataset_name="pizza_sales")
    assert res_b["status"] == "READY"

    viz_b = res_b["visualizations"][0]
    exp_b = build_visualization_explanation(viz_b, dataset_signals=res_b.get("signals"))
    verify_explanation(exp_b, viz_b, expected_status="READY", expected_agg_attribution="User-requested SUM")

    results_table.append({
        "case": case_num,
        "dataset": "pizza_sales.csv",
        "mode": "Stage 8 (User Req)",
        "chart_type": exp_b["chart_type"],
        "columns": f"{exp_b['x_column']} × {exp_b['y_column'] or '-'}",
        "aggregation": exp_b["explanation"]["aggregation"],
        "evidence_count": len(exp_b["explanation"]["evidence"]),
    })
    case_num += 1

    # -----------------------------------------------------------------------
    # Part 3: Invalid & Clarification Fallbacks
    # -----------------------------------------------------------------------
    # Invalid request
    req_inv = parse_visualization_request("Scatter plot of pizza_category and total_price", df_pizza, clf_pizza)
    vreq_inv = validate_visualization_request(df_pizza, clf_pizza, req_inv)
    res_inv = resolve_visualization_request(df_pizza, vreq_inv)
    assert res_inv["status"] == "INVALID"
    exp_inv = build_visualization_explanation(res_inv)
    verify_explanation(exp_inv, {"chart_type": "UNKNOWN", "x_column": "UNKNOWN", "y_column": None}, expected_status="INVALID")

    results_table.append({
        "case": case_num,
        "dataset": "pizza_sales.csv",
        "mode": "Invalid Fallback",
        "chart_type": exp_inv["chart_type"],
        "columns": "- × -",
        "aggregation": exp_inv["explanation"]["aggregation"],
        "evidence_count": len(exp_inv["explanation"]["evidence"]),
    })
    case_num += 1

    # Clarification request
    req_clar = parse_visualization_request("make a chart", df_pizza, clf_pizza)
    vreq_clar = validate_visualization_request(df_pizza, clf_pizza, req_clar)
    res_clar = resolve_visualization_request(df_pizza, vreq_clar)
    assert res_clar["status"] == "NEEDS_CLARIFICATION"
    exp_clar = build_visualization_explanation(res_clar)
    verify_explanation(exp_clar, {"chart_type": "UNKNOWN", "x_column": "UNKNOWN", "y_column": None}, expected_status="NEEDS_CLARIFICATION")

    results_table.append({
        "case": case_num,
        "dataset": "pizza_sales.csv",
        "mode": "Clarification Fallback",
        "chart_type": exp_clar["chart_type"],
        "columns": "- × -",
        "aggregation": exp_clar["explanation"]["aggregation"],
        "evidence_count": len(exp_clar["explanation"]["evidence"]),
    })

    # -----------------------------------------------------------------------
    # Print Results Table
    # -----------------------------------------------------------------------
    print(f"{'#':<3} | {'Dataset':<23} | {'Mode':<22} | {'Chart Type':<10} | {'Columns':<36} | {'Evidence':<8} | Aggregation")
    print("-" * 135)
    for r in results_table:
        print(
            f"{r['case']:<3} | {r['dataset']:<23} | {r['mode']:<22} | {r['chart_type']:<10} | "
            f"{r['columns']:<36} | {r['evidence_count']:<8} | {r['aggregation']}"
        )
    print("-" * 135)
    print("ALL REAL-DATA EXPLANATION VALIDATIONS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

