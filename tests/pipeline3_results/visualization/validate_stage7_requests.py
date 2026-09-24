"""Real-Data Validation Script for Pipeline 3 Stage 7: Visualization Request Intelligence.

Tests natural-language user requests against real LYNX datasets from tests/datasets/valid/
to verify deterministic understanding, validation, and error reporting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from pipelines.pipeline3_results.visualization.column_classifier import classify_columns
from pipelines.pipeline3_results.visualization.request_parser import parse_visualization_request
from pipelines.pipeline3_results.visualization.request_validator import validate_visualization_request

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = REPO_ROOT / "tests" / "datasets" / "valid"

SCENARIOS: list[dict[str, Any]] = [
    # 1. pizza_sales.csv
    {
        "dataset_name": "pizza_sales.csv",
        "load_type": "csv",
        "file_path": DATASETS_DIR / "pizza_sales.csv",
        "queries": [
            ("Show total sales by pizza category", "VALID", "BAR"),
            ("Plot total_price over order_date", "VALID", "LINE"),
            ("Scatter plot of unit_price vs total_price", "VALID", "SCATTER"),
            ("Histogram of total_price", "VALID", "HISTOGRAM"),
            ("Count of pizza_size", "VALID", "COUNT_BAR"),
            ("Show a histogram of pizza_ingredients", "INVALID", None),
            ("Plot order_id against pizza_id", "INVALID", None),
            ("Make me a graph", "NEEDS_CLARIFICATION", None),
            ("Make a scatter plot", "NEEDS_CLARIFICATION", "SCATTER"),
        ],
    },
    # 2. dataset_01_retail_transactions_raw.csv
    {
        "dataset_name": "dataset_01_retail_transactions_raw.csv",
        "load_type": "csv",
        "file_path": DATASETS_DIR / "dataset_01_retail_transactions_raw.csv",
        "queries": [
            ("Show total sales by sales channel", "VALID", "BAR"),
            ("Line graph of total_amount over transaction_date", "VALID", "LINE"),
            ("Plot transaction_id against customer_id", "INVALID", None),
            ("Scatter plot of quantity versus total_amount", "VALID", "SCATTER"),
        ],
    },
    # 3. dataset_02_hr_employees_raw.csv
    {
        "dataset_name": "dataset_02_hr_employees_raw.csv",
        "load_type": "csv",
        "file_path": DATASETS_DIR / "dataset_02_hr_employees_raw.csv",
        "queries": [
            ("Count of employment_status", "VALID", "COUNT_BAR"),
            ("Show leave_balance_days by gender", "VALID", "BAR"),
            ("Histogram of employee_id", "INVALID", None),
            ("Show full_name by leave_balance_days", "INVALID", None),
        ],
    },
    # 4. dataset_07_iot_sensor_raw.csv
    {
        "dataset_name": "dataset_07_iot_sensor_raw.csv",
        "load_type": "csv",
        "file_path": DATASETS_DIR / "dataset_07_iot_sensor_raw.csv",
        "queries": [
            ("Scatter plot of temperature versus vibration_mm_s", "VALID", "SCATTER"),
            ("Average vibration_mm_s by operating_state", "VALID", "BAR"),
            ("Histogram of reading_id", "INVALID", None),
        ],
    },
    # 5. dataset_09_survey_data_raw.csv
    {
        "dataset_name": "dataset_09_survey_data_raw.csv",
        "load_type": "csv",
        "file_path": DATASETS_DIR / "dataset_09_survey_data_raw.csv",
        "queries": [
            ("Count of device_type", "VALID", "COUNT_BAR"),
            ("Line graph over time", "INVALID", None),
            ("Histogram of open_feedback", "INVALID", None),
        ],
    },
    # 6. dataset_11_scientific_research_raw.csv
    {
        "dataset_name": "dataset_11_scientific_research_raw.csv",
        "load_type": "csv",
        "file_path": DATASETS_DIR / "dataset_11_scientific_research_raw.csv",
        "queries": [
            ("Count of calibration_status", "VALID", "COUNT_BAR"),
            ("Plot record_id against study_id", "INVALID", None),
        ],
    },
    # 7. empty_dataset.csv (edge case)
    {
        "dataset_name": "empty_dataset.csv",
        "load_type": "synthetic_empty",
        "file_path": None,
        "queries": [
            ("Show sales by category", "INVALID", None),
            ("Make me a graph", "NEEDS_CLARIFICATION", None),
        ],
    },
    # 8. all_constant_dataset.csv (edge case)
    {
        "dataset_name": "all_constant_dataset.csv",
        "load_type": "synthetic_constant",
        "file_path": None,
        "queries": [
            ("Histogram of const_a", "INVALID", None),
        ],
    },
]


def run_stage7_validation() -> None:
    print("=" * 80)
    print("LYNX PIPELINE 3 — STAGE 7: REAL-DATA REQUEST VALIDATION")
    print("=" * 80)

    total_queries = 0
    passed_queries = 0

    for item in SCENARIOS:
        ds_name = item["dataset_name"]
        print(f"\n--- Dataset: {ds_name} ---")

        if item["load_type"] == "csv":
            df = pd.read_csv(item["file_path"])
        elif item["load_type"] == "synthetic_empty":
            df = pd.DataFrame({"id": pd.Series([], dtype=str), "sales": pd.Series([], dtype=float)})
        elif item["load_type"] == "synthetic_constant":
            df = pd.DataFrame({"const_a": [1] * 20, "const_b": ["fixed"] * 20})
        else:
            raise ValueError(f"Unknown load type: {item['load_type']}")

        classification = classify_columns(df)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns (Measures: {len(classification.measures)}, Categories: {len(classification.categories)}, Temporals: {len(classification.temporals)})")

        for query_text, expected_status, expected_chart in item["queries"]:
            total_queries += 1
            req = parse_visualization_request(query_text, df, classification)
            res = validate_visualization_request(df, classification, req)

            # JSON serializability check
            d = res.to_dict()
            json_str = json.dumps(d)
            reloaded = json.loads(json_str)
            assert reloaded == d, "JSON round-trip mismatch"

            status_ok = (res.status == expected_status)
            chart_ok = (expected_chart is None or res.chart_type == expected_chart)

            if status_ok and chart_ok:
                passed_queries += 1
                flag = "PASS"
            else:
                flag = "FAIL"

            details = f"x={res.x_column}, y={res.y_column}, chart={res.chart_type}"
            if res.errors:
                details += f" | err={res.errors[0]}"
            elif res.clarification_options:
                details += f" | opts={len(res.clarification_options)}"

            print(f"  [{flag}] \"{query_text}\" -> {res.status} ({details})")

    print("\n" + "=" * 80)
    print(f"STAGE 7 REAL-DATA VALIDATION COMPLETE: {passed_queries}/{total_queries} queries verified.")
    print("=" * 80)


if __name__ == "__main__":
    run_stage7_validation()
