"""Pipeline 3 -- Full End-to-End Integration Validation.

Validates the complete Pipeline 3 chain from dataset input to Stage 10 response.
Covers:
  PATH A -- Automatic visualization (Stage 6 + Stage 10)
  PATH B -- Requested visualization (Stage 7->8->9->10)
  NEG 1-5 -- Negative path end-to-end
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipelines.pipeline3_results.visualization.column_classifier import classify_dataframe
from pipelines.pipeline3_results.visualization.explanation import (
    FORBIDDEN_WORDS,
    build_visualization_explanation,
)
from pipelines.pipeline3_results.visualization.feasibility import (
    analyze_visualization_feasibility,
)
from pipelines.pipeline3_results.visualization.integration import build_visualization_payload
from pipelines.pipeline3_results.visualization.request_parser import (
    parse_visualization_request,
)
from pipelines.pipeline3_results.visualization.request_resolver import (
    resolve_visualization_request,
)
from pipelines.pipeline3_results.visualization.request_validator import (
    validate_visualization_request,
)
from pipelines.pipeline3_results.visualization.response_builder import (
    build_visualization_response,
)

_PIZZA_PATH = _PROJECT_ROOT / "tests" / "datasets" / "valid" / "pizza_sales.csv"


def _row(
    path: str,
    dataset: str,
    request: str,
    s7_status: str,
    s8_status: str,
    s10_status: str,
    chart_type: str,
    x_col: str,
    y_col: str,
    aggregation: str,
    score: str,
    explanation_present: str,
    result: str,
    notes: str = "",
) -> dict[str, str]:
    return {
        "path": path,
        "dataset": dataset,
        "request": request,
        "s7_status": s7_status,
        "s8_status": s8_status,
        "s10_status": s10_status,
        "chart_type": chart_type,
        "x_col": x_col,
        "y_col": y_col,
        "aggregation": aggregation,
        "score": score,
        "explanation_present": explanation_present,
        "result": result,
        "notes": notes,
    }


def _check_forbidden(text: str) -> list[str]:
    words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    return sorted(words & FORBIDDEN_WORDS)


def _check_all_forbidden(response: dict[str, Any]) -> list[str]:
    hits: list[str] = []
    for viz in response.get("visualizations", []):
        expl = viz.get("explanation", {})
        for key in ("why_this_chart", "what_it_shows", "aggregation", "data_coverage"):
            val = expl.get(key, "")
            if isinstance(val, str):
                hits.extend(_check_forbidden(val))
        for ev in expl.get("evidence", []):
            if isinstance(ev, str):
                hits.extend(_check_forbidden(ev))
    return list(set(hits))


def _is_json_serializable(obj: Any) -> tuple[bool, str]:
    try:
        json.dumps(obj)
        return True, ""
    except (TypeError, ValueError) as exc:
        return False, str(exc)


def _explanation_present(response: dict[str, Any]) -> bool:
    vizs = response.get("visualizations", [])
    if not vizs:
        return True
    return all(isinstance(v.get("explanation"), dict) and v["explanation"] for v in vizs)


def _score_str(score: Any) -> str:
    if score is None:
        return "None"
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def run_path_a(df: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    failures: list[str] = []
    payload = build_visualization_payload(df, dataset_name="pizza_sales", max_visualizations=5)
    response = build_visualization_response(payload, mode="automatic")

    if response.get("status") != "READY":
        failures.append(f"Expected status=READY, got {response.get('status')!r}")

    vizs = response.get("visualizations", [])
    if not vizs:
        failures.append("No visualizations in automatic response")

    if vizs:
        first_viz = vizs[0]
        cd = first_viz.get("chart_data", {})
        if not cd.get("has_data", False):
            failures.append("chart_data.has_data is False in first visualization")
        if not cd.get("points") and not cd.get("bins"):
            failures.append("chart_data has no points and no bins in first visualization")

        score_val = first_viz.get("score")
        if score_val is None:
            failures.append("Automatic viz score must be float, got None")
        else:
            try:
                float(score_val)
            except (TypeError, ValueError):
                failures.append(f"score is not numeric: {score_val!r}")

        if not _explanation_present(response):
            failures.append("Explanation missing or empty on one or more visualizations")

        bad_words = _check_all_forbidden(response)
        if bad_words:
            failures.append(f"Forbidden words found in explanations: {bad_words}")

    ok, err = _is_json_serializable(response)
    if not ok:
        failures.append(f"Response is not JSON-serializable: {err}")

    first_x = vizs[0]["x_column"] if vizs else "-"
    first_y = vizs[0].get("y_column") or "None"
    first_ct = vizs[0]["chart_type"] if vizs else "-"
    first_agg = vizs[0].get("chart_data", {}).get("aggregation") or "None"
    first_score = _score_str(vizs[0].get("score")) if vizs else "-"

    result_str = "PASS" if not failures else "FAIL"
    row = _row(
        path="A - Automatic",
        dataset="pizza_sales.csv",
        request="(automatic)",
        s7_status="-",
        s8_status="-",
        s10_status=response.get("status", "?"),
        chart_type=first_ct,
        x_col=first_x,
        y_col=first_y,
        aggregation=str(first_agg),
        score=first_score,
        explanation_present="YES" if _explanation_present(response) else "NO",
        result=result_str,
        notes="; ".join(failures),
    )
    return row, failures


def run_path_b(df: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    failures: list[str] = []
    clf = classify_dataframe(df)
    feas = analyze_visualization_feasibility(df, clf)

    request_text = "Median unit_price by pizza_category"
    parsed = parse_visualization_request(request_text, df, clf)
    validated = validate_visualization_request(df, clf, parsed, feas)

    if validated.status != "VALID":
        failures.append(f"Expected Stage 7 status=VALID, got {validated.status!r}")

    resolved = resolve_visualization_request(df, validated, dataset_name="pizza_sales")
    response = build_visualization_response(resolved, mode="requested")

    if resolved.get("status") != "READY":
        failures.append(f"Expected Stage 8 status=READY, got {resolved.get('status')!r}")

    vizs = response.get("visualizations", [])
    if not vizs:
        failures.append("No visualizations in requested response")
        return _row(
            path="B - Requested",
            dataset="pizza_sales.csv",
            request=request_text,
            s7_status=str(validated.status),
            s8_status=resolved.get("status", "?"),
            s10_status=response.get("status", "?"),
            chart_type="-", x_col="-", y_col="-", aggregation="-",
            score="-", explanation_present="NO",
            result="FAIL", notes="; ".join(failures),
        ), failures

    viz = vizs[0]
    cd = viz.get("chart_data", {})

    if viz.get("chart_type") != "BAR":
        failures.append(f"Expected chart_type=BAR, got {viz.get('chart_type')!r}")
    if viz.get("x_column") != "pizza_category":
        failures.append(f"Expected x_column='pizza_category', got {viz.get('x_column')!r}")
    if viz.get("y_column") != "unit_price":
        failures.append(f"Expected y_column='unit_price', got {viz.get('y_column')!r}")
    if str(cd.get("aggregation", "")).upper() != "MEDIAN":
        failures.append(f"Expected aggregation=MEDIAN, got {cd.get('aggregation')!r}")
    if viz.get("score") is not None:
        failures.append(f"Expected score=None for requested viz, got {viz.get('score')!r}")

    expl = viz.get("explanation", {})
    agg_attribution = str(expl.get("aggregation", ""))
    if "user-requested" not in agg_attribution.lower() or "median" not in agg_attribution.lower():
        failures.append(f"Explanation did not identify user-requested MEDIAN aggregation: {agg_attribution!r}")

    bad_words = _check_all_forbidden(response)
    if bad_words:
        failures.append(f"Forbidden words in explanations: {bad_words}")

    ok, err = _is_json_serializable(response)
    if not ok:
        failures.append(f"Response not JSON-serializable: {err}")

    result_str = "PASS" if not failures else "FAIL"
    return _row(
        path="B - Requested",
        dataset="pizza_sales.csv",
        request=request_text,
        s7_status=str(validated.status),
        s8_status=resolved.get("status", "?"),
        s10_status=response.get("status", "?"),
        chart_type=viz.get("chart_type", "?"),
        x_col=viz.get("x_column", "?"),
        y_col=viz.get("y_column") or "None",
        aggregation=str(cd.get("aggregation", "?")),
        score=_score_str(viz.get("score")),
        explanation_present="YES" if _explanation_present(response) else "NO",
        result=result_str,
        notes="; ".join(failures),
    ), failures


def run_neg1_vague(df: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    failures: list[str] = []
    clf = classify_dataframe(df)
    feas = analyze_visualization_feasibility(df, clf)
    parsed = parse_visualization_request("make a chart", df, clf)
    validated = validate_visualization_request(df, clf, parsed, feas)
    resolved = resolve_visualization_request(df, validated, dataset_name="pizza_sales")
    response = build_visualization_response(resolved, mode="requested")

    if validated.status != "NEEDS_CLARIFICATION":
        failures.append(f"Expected NEEDS_CLARIFICATION, got {validated.status}")
    ok, err = _is_json_serializable(response)
    if not ok:
        failures.append(f"Response not JSON-serializable: {err}")

    return _row(
        path="NEG1 - Vague",
        dataset="pizza_sales.csv",
        request="make a chart",
        s7_status=str(validated.status),
        s8_status=resolved.get("status", "?"),
        s10_status=response.get("status", "?"),
        chart_type="-", x_col="-", y_col="-", aggregation="-",
        score="-",
        explanation_present="YES" if _explanation_present(response) else "NO",
        result="PASS" if not failures else "FAIL",
        notes="; ".join(failures),
    ), failures


def run_neg2_scatter_categorical(df: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    failures: list[str] = []
    clf = classify_dataframe(df)
    feas = analyze_visualization_feasibility(df, clf)
    parsed = parse_visualization_request("Scatter plot of pizza_category and unit_price", df, clf)
    validated = validate_visualization_request(df, clf, parsed, feas)
    resolved = resolve_visualization_request(df, validated, dataset_name="pizza_sales")
    response = build_visualization_response(resolved, mode="requested")

    if validated.status != "INVALID":
        failures.append(f"Expected INVALID for scatter+categorical, got {validated.status}")
    ok, err = _is_json_serializable(response)
    if not ok:
        failures.append(f"Response not JSON-serializable: {err}")

    return _row(
        path="NEG2 - Scatter+Cat",
        dataset="pizza_sales.csv",
        request="Scatter: pizza_category + unit_price",
        s7_status=str(validated.status),
        s8_status=resolved.get("status", "?"),
        s10_status=response.get("status", "?"),
        chart_type="-", x_col="-", y_col="-", aggregation="-",
        score="-",
        explanation_present="YES" if _explanation_present(response) else "NO",
        result="PASS" if not failures else "FAIL",
        notes="; ".join(failures),
    ), failures


def run_neg3_empty_dataset() -> tuple[dict[str, str], list[str]]:
    failures: list[str] = []
    empty_df = pd.DataFrame({"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype="object")})
    payload = build_visualization_payload(empty_df, dataset_name="empty")
    response = build_visualization_response(payload, mode="automatic")

    if response.get("status") != "NO_MEANINGFUL_VISUALIZATION":
        failures.append(f"Expected NO_MEANINGFUL_VISUALIZATION, got {response.get('status')!r}")
    ok, err = _is_json_serializable(response)
    if not ok:
        failures.append(f"Response not JSON-serializable: {err}")

    return _row(
        path="NEG3 - Empty DF",
        dataset="(empty)",
        request="(automatic)",
        s7_status="-", s8_status="-",
        s10_status=response.get("status", "?"),
        chart_type="-", x_col="-", y_col="-", aggregation="-",
        score="-",
        explanation_present="YES" if _explanation_present(response) else "NO",
        result="PASS" if not failures else "FAIL",
        notes="; ".join(failures),
    ), failures


def run_neg4_two_row_dataset(df: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    failures: list[str] = []
    two_row = df.head(2).copy()
    payload = build_visualization_payload(two_row, dataset_name="two_row")
    response = build_visualization_response(payload, mode="automatic")

    if response.get("status") != "NO_MEANINGFUL_VISUALIZATION":
        failures.append(f"Expected NO_MEANINGFUL_VISUALIZATION, got {response.get('status')!r}")
    ok, err = _is_json_serializable(response)
    if not ok:
        failures.append(f"Response not JSON-serializable: {err}")

    return _row(
        path="NEG4 - Two Rows",
        dataset="pizza (head 2)",
        request="(automatic)",
        s7_status="-", s8_status="-",
        s10_status=response.get("status", "?"),
        chart_type="-", x_col="-", y_col="-", aggregation="-",
        score="-",
        explanation_present="YES" if _explanation_present(response) else "NO",
        result="PASS" if not failures else "FAIL",
        notes="; ".join(failures),
    ), failures


def run_neg5_median_scatter(df: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    failures: list[str] = []
    clf = classify_dataframe(df)
    feas = analyze_visualization_feasibility(df, clf)
    parsed = parse_visualization_request(
        "scatter plot of unit_price vs total_price with median aggregation", df, clf
    )
    validated = validate_visualization_request(df, clf, parsed, feas)
    resolved = resolve_visualization_request(df, validated, dataset_name="pizza_sales")
    response = build_visualization_response(resolved, mode="requested")

    if resolved.get("status") != "INVALID":
        failures.append(f"Expected Stage 8 INVALID for median-on-scatter; got {resolved.get('status')!r}")
    ok, err = _is_json_serializable(response)
    if not ok:
        failures.append(f"Response not JSON-serializable: {err}")

    return _row(
        path="NEG5 - Median+Scatter",
        dataset="pizza_sales.csv",
        request="scatter unit_price vs total_price (median)",
        s7_status=str(validated.status),
        s8_status=resolved.get("status", "?"),
        s10_status=response.get("status", "?"),
        chart_type="-", x_col="-", y_col="-", aggregation="-",
        score="-",
        explanation_present="YES" if _explanation_present(response) else "NO",
        result="PASS" if not failures else "FAIL",
        notes="; ".join(failures),
    ), failures


def main() -> int:
    print("\n" + "=" * 80)
    print("Pipeline 3 -- Full End-to-End Integration Validation")
    print("=" * 80)

    df = pd.read_csv(_PIZZA_PATH)
    print(f"\nLoaded pizza_sales.csv: {df.shape[0]:,} rows x {df.shape[1]} cols\n")

    all_rows: list[dict[str, str]] = []
    all_failures: list[tuple[str, list[str]]] = []

    for name, runner, arg in [
        ("PATH A", run_path_a, df),
        ("PATH B", run_path_b, df),
        ("NEG 1", run_neg1_vague, df),
        ("NEG 2", run_neg2_scatter_categorical, df),
        ("NEG 3", run_neg3_empty_dataset, None),
        ("NEG 4", run_neg4_two_row_dataset, df),
        ("NEG 5", run_neg5_median_scatter, df),
    ]:
        row, fails = runner(arg) if arg is not None else runner()
        all_rows.append(row)
        if fails:
            all_failures.append((name, fails))
        print(f"[{name}] Result: {row['result']}")

    total = len(all_rows)
    passed = sum(1 for r in all_rows if r["result"] == "PASS")
    failed = total - passed
    print(f"\nPipeline 3 End-to-End: {passed}/{total} passed, {failed} failed\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())