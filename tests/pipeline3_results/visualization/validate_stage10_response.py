"""Real-data validation for Pipeline 3 Stage 10 -- Unified Visualization Response Contract.

Runs build_visualization_response() against representative real datasets
using both automatic (Stage 6) and requested (Stage 8) paths.

Datasets used:
 1. pizza_sales.csv                -- large retail dataset
 2. dataset_09_survey_data_raw.csv -- categorical-heavy survey data
 3. dataset_07_iot_sensor_raw.csv  -- numeric/temporal IoT signals
 4. TabArena Airfoil               -- all-numeric scientific dataset
 5. TabArena Used Cars             -- mixed numeric+categorical automotive
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

from pipelines.pipeline3_results.visualization.column_classifier import classify_dataframe
from pipelines.pipeline3_results.visualization.feasibility import analyze_visualization_feasibility
from pipelines.pipeline3_results.visualization.integration import build_visualization_payload
from pipelines.pipeline3_results.visualization.request_parser import parse_visualization_request
from pipelines.pipeline3_results.visualization.request_resolver import resolve_visualization_request
from pipelines.pipeline3_results.visualization.request_validator import validate_visualization_request
from pipelines.pipeline3_results.visualization.response_builder import build_visualization_response

DATASETS = {
    "pizza_sales": _PROJECT_ROOT / "tests" / "datasets" / "valid" / "pizza_sales.csv",
    "survey": _PROJECT_ROOT / "tests" / "datasets" / "valid" / "dataset_09_survey_data_raw.csv",
    "iot_sensor": _PROJECT_ROOT / "tests" / "datasets" / "valid" / "dataset_07_iot_sensor_raw.csv",
    "airfoil": _PROJECT_ROOT / "benchmarks" / "tabarena" / "pipeline1_outputs" / "363612" / "cleaned_dataset.csv",
    "used_cars": _PROJECT_ROOT / "benchmarks" / "tabarena" / "pipeline1_outputs" / "363615" / "cleaned_dataset.csv",
}

_REQUIRED_KEYS = {"mode", "status", "dataset_name", "visualization_count", "visualizations", "signals"}
_REQUIRED_EXPLANATION_KEYS = {"why_this_chart", "what_it_shows", "aggregation", "data_coverage", "evidence"}


def load(name: str) -> pd.DataFrame | None:
    path = DATASETS[name]
    if not path.exists():
        print(f"  [SKIP] {name}: file not found at {path}")
        return None
    try:
        df = pd.read_csv(path, nrows=2000)
        print(f"  [LOAD] {name}: {df.shape[0]} rows x {df.shape[1]} cols")
        return df
    except Exception as exc:
        print(f"  [SKIP] {name}: could not load - {exc}")
        return None


def _assert_response_structure(response: dict[str, Any], mode: str, case_label: str) -> None:
    missing_keys = _REQUIRED_KEYS - response.keys()
    assert not missing_keys, f"[{case_label}] Missing top-level keys: {missing_keys}"
    assert response["mode"] == mode, f"[{case_label}] mode mismatch"
    assert isinstance(response["visualizations"], list), f"[{case_label}] visualizations not a list"
    for viz in response["visualizations"]:
        assert "explanation" in viz, f"[{case_label}] explanation missing from viz rank={viz.get('rank')}"
        expl_missing = _REQUIRED_EXPLANATION_KEYS - viz["explanation"].keys()
        assert not expl_missing, f"[{case_label}] explanation missing keys: {expl_missing}"
    json.dumps(response)


def run_automatic_case(name: str, df: pd.DataFrame) -> bool:
    try:
        payload = build_visualization_payload(df, dataset_name=name, max_visualizations=5)
        response = build_visualization_response(payload, mode="automatic", dataset_signals=["real-data validation"])
        _assert_response_structure(response, "automatic", f"auto:{name}")
        viz_count = len(response["visualizations"])
        print(f"  [PASS] auto:{name} - status={response['status']}, {viz_count} visualization(s) with explanation")
        return True
    except Exception as exc:
        print(f"  [FAIL] auto:{name} - {exc}")
        return False


def run_requested_case(name: str, df: pd.DataFrame, request_text: str) -> bool:
    try:
        classification = classify_dataframe(df)
        feasibility = analyze_visualization_feasibility(df, classification)
        parsed = parse_visualization_request(request_text, df)
        validated = validate_visualization_request(df, classification, parsed, feasibility)
        resolved = resolve_visualization_request(df, validated, dataset_name=name)
        response = build_visualization_response(resolved, mode="requested")
        _assert_response_structure(response, "requested", f"req:{name}")
        status = response["status"]
        viz_count = len(response["visualizations"])
        print(f"  [PASS] req:{name} - status={status}, {viz_count} viz(s), request={request_text!r}")
        return True
    except Exception as exc:
        print(f"  [FAIL] req:{name} ({request_text!r}) - {exc}")
        return False


def main() -> None:
    passed = 0
    failed = 0
    total = 0

    print("\n" + "=" * 70)
    print("Pipeline 3 Stage 10 - Real-Data Validation")
    print("=" * 70)

    print("\n[1] pizza_sales - automatic")
    df_pizza = load("pizza_sales")
    if df_pizza is not None:
        total += 1
        if run_automatic_case("pizza_sales", df_pizza):
            passed += 1
        else:
            failed += 1

    print("\n[2] pizza_sales - requested: bar chart pizza_category")
    if df_pizza is not None:
        total += 1
        if run_requested_case("pizza_sales", df_pizza, "show a bar chart of pizza_category"):
            passed += 1
        else:
            failed += 1

    print("\n[3] pizza_sales - requested: total_price by pizza_category")
    if df_pizza is not None:
        total += 1
        if run_requested_case("pizza_sales", df_pizza, "show total_price by pizza_category"):
            passed += 1
        else:
            failed += 1

    print("\n[4] survey - automatic")
    df_survey = load("survey")
    if df_survey is not None:
        total += 1
        if run_automatic_case("survey", df_survey):
            passed += 1
        else:
            failed += 1

    print("\n[5] survey - requested categorical bar")
    if df_survey is not None:
        cat_cols = [c for c in df_survey.columns if df_survey[c].dtype == object]
        if cat_cols:
            total += 1
            req = f"show bar chart of {cat_cols[0]}"
            if run_requested_case("survey", df_survey, req):
                passed += 1
            else:
                failed += 1

    print("\n[6] iot_sensor - automatic")
    df_iot = load("iot_sensor")
    if df_iot is not None:
        total += 1
        if run_automatic_case("iot_sensor", df_iot):
            passed += 1
        else:
            failed += 1

    print("\n[7] iot_sensor - requested: histogram")
    if df_iot is not None:
        num_cols = [c for c in df_iot.columns if pd.api.types.is_numeric_dtype(df_iot[c])]
        if num_cols:
            total += 1
            req = f"show histogram of {num_cols[0]}"
            if run_requested_case("iot_sensor", df_iot, req):
                passed += 1
            else:
                failed += 1

    print("\n[8] airfoil (TabArena) - automatic")
    df_airfoil = load("airfoil")
    if df_airfoil is not None:
        total += 1
        if run_automatic_case("airfoil", df_airfoil):
            passed += 1
        else:
            failed += 1

    print("\n[9] airfoil (TabArena) - requested: scatter")
    if df_airfoil is not None:
        num_cols = [c for c in df_airfoil.columns if pd.api.types.is_numeric_dtype(df_airfoil[c])]
        if len(num_cols) >= 2:
            total += 1
            req = f"scatter plot {num_cols[0]} vs {num_cols[1]}"
            if run_requested_case("airfoil", df_airfoil, req):
                passed += 1
            else:
                failed += 1

    print("\n[10] used_cars (TabArena) - automatic")
    df_cars = load("used_cars")
    if df_cars is not None:
        total += 1
        if run_automatic_case("used_cars", df_cars):
            passed += 1
        else:
            failed += 1

    print("\n[11] used_cars (TabArena) - requested: bar chart")
    if df_cars is not None:
        cat_cols = [c for c in df_cars.columns if df_cars[c].dtype == object and df_cars[c].nunique() < 30]
        num_cols = [c for c in df_cars.columns if pd.api.types.is_numeric_dtype(df_cars[c])]
        if cat_cols and num_cols:
            total += 1
            req = f"show {num_cols[0]} by {cat_cols[0]}"
            if run_requested_case("used_cars", df_cars, req):
                passed += 1
            else:
                failed += 1

    print("\n[12] INVALID status passthrough")
    total += 1
    try:
        invalid_payload: dict[str, Any] = {
            "status": "INVALID",
            "dataset_name": "test",
            "visualization_count": 0,
            "visualizations": [],
            "signals": ["Column 'xyz' not found."],
            "errors": ["Column 'xyz' does not exist."],
        }
        response = build_visualization_response(invalid_payload, mode="requested")
        assert response["status"] == "INVALID"
        assert response["mode"] == "requested"
        assert "errors" in response
        json.dumps(response)
        print("  [PASS] INVALID status passthrough")
        passed += 1
    except Exception as exc:
        print(f"  [FAIL] INVALID status passthrough - {exc}")
        failed += 1

    print("\n[13] NEEDS_CLARIFICATION passthrough")
    total += 1
    try:
        clarif_payload: dict[str, Any] = {
            "status": "NEEDS_CLARIFICATION",
            "dataset_name": "test",
            "visualization_count": 0,
            "visualizations": [],
            "signals": ["Multiple columns match."],
            "clarification_options": ["Use 'price' or 'unit_price'?"],
        }
        response = build_visualization_response(clarif_payload, mode="requested")
        assert response["status"] == "NEEDS_CLARIFICATION"
        assert "clarification_options" in response
        json.dumps(response)
        print("  [PASS] NEEDS_CLARIFICATION passthrough")
        passed += 1
    except Exception as exc:
        print(f"  [FAIL] NEEDS_CLARIFICATION - {exc}")
        failed += 1

    print("\n[14] Score invariance - automatic score preserved")
    total += 1
    try:
        score_payload: dict[str, Any] = {
            "status": "READY",
            "dataset_name": "score_test",
            "visualization_count": 1,
            "visualizations": [{
                "rank": 1,
                "chart_type": "COUNT_BAR",
                "x_column": "category",
                "y_column": None,
                "score": 0.888,
                "chart_data": {
                    "chart_type": "COUNT_BAR",
                    "x_column": "category",
                    "y_column": None,
                    "aggregation": "count",
                    "points": [{"x": "A", "y": 50}, {"x": "B", "y": 30}],
                    "bins": [],
                    "signals": [],
                    "usable_row_count": 80,
                    "sampled": False,
                    "has_data": True,
                },
                "signals": [],
            }],
            "signals": [],
        }
        response = build_visualization_response(score_payload, mode="automatic")
        assert response["visualizations"][0]["score"] == 0.888, "Score was altered"
        print(f"  [PASS] Automatic score preserved: {response['visualizations'][0]['score']}")
        passed += 1
    except Exception as exc:
        print(f"  [FAIL] Score invariance - {exc}")
        failed += 1

    print("\n[15] Score invariance - requested score=None stays None")
    total += 1
    try:
        none_score_payload: dict[str, Any] = {
            "status": "READY",
            "dataset_name": "none_score_test",
            "visualization_count": 1,
            "visualizations": [{
                "rank": 1,
                "chart_type": "COUNT_BAR",
                "x_column": "category",
                "y_column": None,
                "score": None,
                "chart_data": {
                    "chart_type": "COUNT_BAR",
                    "x_column": "category",
                    "y_column": None,
                    "aggregation": "count",
                    "points": [{"x": "A", "y": 50}, {"x": "B", "y": 30}],
                    "bins": [],
                    "signals": [],
                    "usable_row_count": 80,
                    "sampled": False,
                    "has_data": True,
                },
                "signals": [],
            }],
            "signals": [],
        }
        response = build_visualization_response(none_score_payload, mode="requested")
        assert response["visualizations"][0]["score"] is None, "None score was replaced"
        print("  [PASS] Requested score=None preserved")
        passed += 1
    except Exception as exc:
        print(f"  [FAIL] None score invariance - {exc}")
        failed += 1

    print("\n[16] Visualization ordering preserved")
    total += 1
    try:
        ordering_payload: dict[str, Any] = {
            "status": "READY",
            "dataset_name": "order_test",
            "visualization_count": 3,
            "visualizations": [
                {
                    "rank": 1, "chart_type": "COUNT_BAR", "x_column": "cat", "y_column": None,
                    "score": 0.999,
                    "chart_data": {
                        "chart_type": "COUNT_BAR", "x_column": "cat", "y_column": None,
                        "aggregation": "count",
                        "points": [{"x": "A", "y": 10}, {"x": "B", "y": 5}],
                        "bins": [], "signals": [], "usable_row_count": 15, "sampled": False, "has_data": True,
                    }, "signals": [],
                },
                {
                    "rank": 2, "chart_type": "BAR", "x_column": "cat", "y_column": "val",
                    "score": 0.95,
                    "chart_data": {
                        "chart_type": "BAR", "x_column": "cat", "y_column": "val",
                        "aggregation": "sum",
                        "points": [{"x": "A", "y": 100}, {"x": "B", "y": 50}],
                        "bins": [], "signals": ["aggregated using sum"], "usable_row_count": 15, "sampled": False, "has_data": True,
                    }, "signals": [],
                },
                {
                    "rank": 3, "chart_type": "HISTOGRAM", "x_column": "val", "y_column": None,
                    "score": 0.80,
                    "chart_data": {
                        "chart_type": "HISTOGRAM", "x_column": "val", "y_column": None,
                        "aggregation": None,
                        "points": [],
                        "bins": [
                            {"bin_start": 0, "bin_end": 50, "count": 10},
                            {"bin_start": 50, "bin_end": 100, "count": 5},
                        ],
                        "signals": [], "usable_row_count": 15, "sampled": False, "has_data": True,
                    }, "signals": [],
                },
            ],
            "signals": [],
        }
        response = build_visualization_response(ordering_payload, mode="automatic")
        ranks = [v["rank"] for v in response["visualizations"]]
        assert ranks == [1, 2, 3], f"Ordering changed: {ranks}"
        print(f"  [PASS] Ordering preserved: {ranks}")
        passed += 1
    except Exception as exc:
        print(f"  [FAIL] Ordering preservation - {exc}")
        failed += 1

    print("\n" + "=" * 70)
    print(f"Stage 10 Real-Data Validation: {passed}/{total} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()