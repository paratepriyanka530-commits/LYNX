"""Real-data validation for Pipeline 3 Stage 10 -- Results Service Composition Layer.

Validates build_visualization_results() across both Automatic and Requested modes
using real heterogeneous datasets from valid/ and benchmarks/tabarena/.

Datasets & Scenarios:
Automatic Mode:
 1. pizza_sales.csv (retail transactions)
 2. dataset_01_retail_transactions_raw.csv (e-commerce transactions)
 3. dataset_07_iot_sensor_raw.csv (sensor telemetry)
 4. dataset_09_survey_data_raw.csv (survey feedback)

Requested Mode:
 1. pizza_sales.csv: "Median unit_price by pizza_category"
 2. pizza_sales.csv: "Line chart of total_price over order_date"
 3. TabArena task_363612: "Scatter plot of attack-angle and scaled-sound-pressure"
 4. TabArena task_363615: "Average price by model"
 5. TabArena task_363613: "Count of ResourceApproved"

Edge Cases:
 - "make a chart" -> NEEDS_CLARIFICATION (no chart generation)
 - "scatter plot of pizza_category and pizza_size" -> INVALID (no chart generation)
 - empty dataset -> NO_MEANINGFUL_VISUALIZATION
 - constant dataset -> NO_MEANINGFUL_VISUALIZATION
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipelines.pipeline3_results.visualization.results_service import (
    build_visualization_results,
)

DATASETS = {
    "pizza_sales": _PROJECT_ROOT / "tests" / "datasets" / "valid" / "pizza_sales.csv",
    "retail": _PROJECT_ROOT / "tests" / "datasets" / "valid" / "dataset_01_retail_transactions_raw.csv",
    "iot": _PROJECT_ROOT / "tests" / "datasets" / "valid" / "dataset_07_iot_sensor_raw.csv",
    "survey": _PROJECT_ROOT / "tests" / "datasets" / "valid" / "dataset_09_survey_data_raw.csv",
    "task_363612": _PROJECT_ROOT / "benchmarks" / "tabarena" / "pipeline1_outputs" / "363612" / "cleaned_dataset.csv",
    "task_363615": _PROJECT_ROOT / "benchmarks" / "tabarena" / "pipeline1_outputs" / "363615" / "cleaned_dataset.csv",
    "task_363613": _PROJECT_ROOT / "benchmarks" / "tabarena" / "pipeline1_outputs" / "363613" / "cleaned_dataset.csv",
}


def load_dataset(name: str, max_rows: int = 2500) -> pd.DataFrame | None:
    path = DATASETS[name]
    if not path.exists():
        print(f"  [SKIP] {name}: file not found at {path}")
        return None
    try:
        df = pd.read_csv(path, nrows=max_rows)
        print(f"  [LOAD] {name}: {df.shape[0]} rows x {df.shape[1]} cols")
        return df
    except Exception as exc:
        print(f"  [ERROR] {name}: load failed - {exc}")
        return None


def _assert_json_serializable(res: dict[str, Any], label: str) -> None:
    try:
        dumped = json.dumps(res)
        reloaded = json.loads(dumped)
        assert reloaded["status"] == res["status"], f"[{label}] status mismatch after JSON roundtrip"
    except Exception as exc:
        raise AssertionError(f"[{label}] JSON serialization failed: {exc}") from exc


def main() -> int:
    print("=" * 80)
    print("PIPELINE 3 -- STAGE 10 REAL-DATA VALIDATION (results_service)")
    print("=" * 80)

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    # -----------------------------------------------------------------------
    # 1. Automatic Mode Validations
    # -----------------------------------------------------------------------
    print("\n--- SECTION 1: Automatic Mode Validations ---")
    auto_datasets = ["pizza_sales", "retail", "iot", "survey"]

    for ds_name in auto_datasets:
        total_tests += 1
        df = load_dataset(ds_name)
        if df is None:
            failed_tests += 1
            continue

        try:
            res = build_visualization_results(
                dataframe=df,
                request_text=None,
                dataset_name=ds_name,
                max_visualizations=5,
            )
            assert res["mode"] == "automatic", "mode must be 'automatic'"
            assert res["status"] in ("READY", "NO_MEANINGFUL_VISUALIZATION"), f"unexpected status: {res['status']}"
            assert res["dataset_name"] == ds_name
            assert len(res["visualizations"]) == res["visualization_count"]

            # Verify explanations and scores for automatic results
            for viz in res["visualizations"]:
                assert "explanation" in viz, "explanation missing from automatic visualization"
                assert "why_this_chart" in viz["explanation"]
                assert "what_it_shows" in viz["explanation"]
                assert viz["score"] is not None, "automatic visualization score must not be None"
                assert "chart_data" in viz

            _assert_json_serializable(res, f"auto_{ds_name}")
            print(f"  [PASS] Automatic: {ds_name} -> status={res['status']}, {len(res['visualizations'])} visualizations with explanations & scores.")
            passed_tests += 1
        except Exception as exc:
            print(f"  [FAIL] Automatic: {ds_name} -> {exc}")
            failed_tests += 1

    # -----------------------------------------------------------------------
    # 2. Requested Mode Validations
    # -----------------------------------------------------------------------
    print("\n--- SECTION 2: Requested Mode Validations ---")
    requested_cases = [
        ("pizza_sales", "Median unit_price by pizza_category", "BAR", "pizza_category", "unit_price", "median"),
        ("pizza_sales", "Line chart of total_price over order_date", "LINE", "order_date", "total_price", None),
        ("task_363612", "Scatter plot of attack-angle and scaled-sound-pressure", "SCATTER", "attack-angle", "scaled-sound-pressure", None),
        ("task_363615", "Average price by model", "BAR", "model", "price", "mean"),
        ("task_363613", "Count of ResourceApproved", "COUNT_BAR", "ResourceApproved", None, "count"),
    ]

    for ds_name, query, exp_chart_type, exp_x, exp_y, exp_agg in requested_cases:
        total_tests += 1
        df = load_dataset(ds_name)
        if df is None:
            failed_tests += 1
            continue

        try:
            res = build_visualization_results(
                dataframe=df,
                request_text=query,
                dataset_name=ds_name,
            )
            assert res["mode"] == "requested", "mode must be 'requested'"
            assert res["request_text"] == query
            assert res["status"] == "READY", f"expected READY, got {res['status']}"
            assert res["visualization_count"] == 1
            assert len(res["visualizations"]) == 1

            viz = res["visualizations"][0]
            assert viz["score"] is None, "score must be None for requested visualization"
            assert viz["chart_type"] == exp_chart_type, f"expected chart_type={exp_chart_type}, got {viz['chart_type']}"
            assert viz["x_column"] == exp_x, f"expected x_column={exp_x}, got {viz['x_column']}"
            if exp_y is not None:
                assert viz["y_column"] == exp_y, f"expected y_column={exp_y}, got {viz['y_column']}"

            cd = viz["chart_data"]
            assert cd["has_data"] is True
            if exp_agg is not None:
                assert str(cd["aggregation"]).lower() == exp_agg.lower(), f"expected agg={exp_agg}, got {cd['aggregation']}"

            assert "explanation" in viz
            assert "why_this_chart" in viz["explanation"]
            assert "what_it_shows" in viz["explanation"]

            _assert_json_serializable(res, f"req_{ds_name}")
            print(f"  [PASS] Requested: {ds_name} ('{query}') -> READY, {viz['chart_type']}(x={exp_x}, y={exp_y}), score=None, explanation attached.")
            passed_tests += 1
        except Exception as exc:
            print(f"  [FAIL] Requested: {ds_name} ('{query}') -> {exc}")
            failed_tests += 1

    # -----------------------------------------------------------------------
    # 3. Edge Case Validations (Clarification, Invalid, Empty, Constant)
    # -----------------------------------------------------------------------
    print("\n--- SECTION 3: Edge Case Validations ---")
    df_pizza = load_dataset("pizza_sales", max_rows=500)

    # 3.1: "make a chart" -> NEEDS_CLARIFICATION
    total_tests += 1
    try:
        res_clarif = build_visualization_results(df_pizza, "make a chart")
        assert res_clarif["status"] == "NEEDS_CLARIFICATION"
        assert res_clarif["visualizations"] == []
        assert "clarification_options" in res_clarif
        assert len(res_clarif["clarification_options"]) > 0
        _assert_json_serializable(res_clarif, "clarif")
        print("  [PASS] 'make a chart' -> NEEDS_CLARIFICATION with options, 0 charts generated.")
        passed_tests += 1
    except Exception as exc:
        print(f"  [FAIL] 'make a chart' -> {exc}")
        failed_tests += 1

    # 3.2: Invalid scatter request -> INVALID
    total_tests += 1
    try:
        res_inv = build_visualization_results(df_pizza, "scatter plot of pizza_category and pizza_size")
        assert res_inv["status"] == "INVALID"
        assert res_inv["visualizations"] == []
        assert "errors" in res_inv
        assert len(res_inv["errors"]) > 0
        _assert_json_serializable(res_inv, "invalid_scatter")
        print("  [PASS] Invalid scatter request -> INVALID with errors, 0 charts generated.")
        passed_tests += 1
    except Exception as exc:
        print(f"  [FAIL] Invalid scatter request -> {exc}")
        failed_tests += 1

    # 3.3: Empty dataset -> NO_MEANINGFUL_VISUALIZATION
    total_tests += 1
    try:
        empty_df = pd.DataFrame()
        res_empty_auto = build_visualization_results(empty_df, None)
        assert res_empty_auto["status"] == "NO_MEANINGFUL_VISUALIZATION"
        assert res_empty_auto["visualizations"] == []

        res_empty_req = build_visualization_results(empty_df, "bar chart of a by b")
        assert res_empty_req["status"] == "NO_MEANINGFUL_VISUALIZATION"
        assert res_empty_req["visualizations"] == []
        _assert_json_serializable(res_empty_auto, "empty_auto")
        _assert_json_serializable(res_empty_req, "empty_req")
        print("  [PASS] Empty dataset -> NO_MEANINGFUL_VISUALIZATION (both modes).")
        passed_tests += 1
    except Exception as exc:
        print(f"  [FAIL] Empty dataset -> {exc}")
        failed_tests += 1

    # 3.4: Constant dataset -> NO_MEANINGFUL_VISUALIZATION
    total_tests += 1
    try:
        const_df = pd.DataFrame({"cat": ["X", "X", "X", "X"], "val": [5.0, 5.0, 5.0, 5.0]})
        res_const = build_visualization_results(const_df, None)
        assert res_const["status"] == "NO_MEANINGFUL_VISUALIZATION"
        assert res_const["visualizations"] == []
        _assert_json_serializable(res_const, "const_auto")
        print("  [PASS] Constant dataset -> NO_MEANINGFUL_VISUALIZATION in automatic mode.")
        passed_tests += 1
    except Exception as exc:
        print(f"  [FAIL] Constant dataset -> {exc}")
        failed_tests += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"STAGE 10 REAL-DATA VALIDATION SUMMARY: {passed_tests}/{total_tests} passed, {failed_tests} failed.")
    print("=" * 80)

    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    sys.exit(main())