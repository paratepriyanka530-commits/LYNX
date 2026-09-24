"""Tests for Visualization Integration Contract — Pipeline 3, Stage 6.

Validates that build_visualization_payload produces a frontend-safe,
strictly JSON-serializable dictionary that faithfully preserves Stage 5
ordering, scoring, chart data, and factual audit signals without modification.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

import numpy as np
import pandas as pd

from pipelines.pipeline3_results.visualization import (
    build_visualization_payload,
    build_visualization_result,
)


def _make_realistic_df() -> pd.DataFrame:
    """Create a realistic sample DataFrame for testing visualization integration."""
    return pd.DataFrame({
        "order_id": [f"ORD-{i:04d}" for i in range(1, 21)],
        "category": ["Electronics", "Clothing", "Food", "Books"] * 5,
        "region": ["North", "South", "East", "West"] * 5,
        "sales_amount": [
            120.5, 45.0, 12.99, 89.0, 210.0, 55.5, 8.50, 115.0,
            99.9, 34.0, 15.0, 78.5, 310.0, 62.0, 9.99, 140.0,
            150.0, 48.0, 22.5, 95.0,
        ],
        "quantity": [1, 2, 3, 1, 4, 2, 1, 3, 2, 1, 5, 2, 4, 3, 1, 2, 3, 2, 4, 1],
        "order_date": [
            f"2023-01-{i:02d}" for i in range(1, 21)
        ],
    })


def _assert_json_safe_types_only(obj: Any, path: str = "root") -> None:
    """Recursively assert that an object contains only JSON-primitive types."""
    allowed_types = (dict, list, str, int, float, bool, type(None))
    if not isinstance(obj, allowed_types):
        raise AssertionError(
            f"Disallowed non-JSON type at {path}: {type(obj).__name__} (value: {obj!r})"
        )
    # Reject numpy types that might subclass Python primitives (e.g. np.bool_, np.int64)
    if isinstance(obj, (np.generic, np.ndarray)):
        raise AssertionError(
            f"Numpy type found at {path}: {type(obj).__name__} (value: {obj!r})"
        )
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise AssertionError(f"Non-string dictionary key at {path}: {type(k).__name__} ({k!r})")
            _assert_json_safe_types_only(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _assert_json_safe_types_only(item, f"{path}[{idx}]")


class TestReadyPayload(unittest.TestCase):
    """Test successful payload generation for a viable dataset."""

    def test_ready_status_and_top_level_keys(self) -> None:
        df = _make_realistic_df()
        payload = build_visualization_payload(df, dataset_name="sales_data", max_visualizations=5)

        self.assertIsInstance(payload, dict)
        expected_keys = {"status", "dataset_name", "visualization_count", "visualizations", "signals"}
        self.assertEqual(set(payload.keys()), expected_keys)

        self.assertEqual(payload["status"], "READY")
        self.assertEqual(payload["dataset_name"], "sales_data")
        self.assertGreater(payload["visualization_count"], 0)
        self.assertLessEqual(payload["visualization_count"], 5)
        self.assertEqual(len(payload["visualizations"]), payload["visualization_count"])
        self.assertIsInstance(payload["signals"], list)
        self.assertGreater(len(payload["signals"]), 0)

    def test_visualization_items_structure(self) -> None:
        df = _make_realistic_df()
        payload = build_visualization_payload(df, max_visualizations=3)

        expected_item_keys = {"rank", "chart_type", "x_column", "y_column", "score", "chart_data", "signals"}
        expected_chart_data_keys = {
            "chart_type", "x_column", "y_column", "aggregation",
            "points", "bins", "signals", "usable_row_count", "sampled", "has_data",
        }

        for item in payload["visualizations"]:
            self.assertIsInstance(item, dict)
            self.assertEqual(set(item.keys()), expected_item_keys)
            self.assertIsInstance(item["rank"], int)
            self.assertIsInstance(item["chart_type"], str)
            self.assertIsInstance(item["x_column"], str)
            self.assertTrue(item["y_column"] is None or isinstance(item["y_column"], str))
            self.assertIsInstance(item["score"], float)
            self.assertGreaterEqual(item["score"], 0.0)
            self.assertLessEqual(item["score"], 1.0)
            self.assertIsInstance(item["signals"], list)

            # Validate chart_data sub-dict
            cd = item["chart_data"]
            self.assertIsInstance(cd, dict)
            self.assertEqual(set(cd.keys()), expected_chart_data_keys)
            self.assertEqual(cd["chart_type"], item["chart_type"])
            self.assertEqual(cd["x_column"], item["x_column"])
            self.assertEqual(cd["y_column"], item["y_column"])
            self.assertIsInstance(cd["points"], list)
            self.assertIsInstance(cd["bins"], list)
            self.assertIsInstance(cd["signals"], list)
            self.assertIsInstance(cd["usable_row_count"], int)
            self.assertIsInstance(cd["sampled"], bool)
            self.assertIsInstance(cd["has_data"], bool)
            self.assertTrue(cd["has_data"])


class TestNoMeaningfulVisualization(unittest.TestCase):
    """Test clean handling of degenerate or unvisualizable datasets."""

    def test_constant_columns_dataset(self) -> None:
        df = pd.DataFrame({"c1": [42] * 10, "c2": ["fixed"] * 10})
        payload = build_visualization_payload(df, dataset_name="constant_df")

        self.assertEqual(payload["status"], "NO_MEANINGFUL_VISUALIZATION")
        self.assertEqual(payload["dataset_name"], "constant_df")
        self.assertEqual(payload["visualization_count"], 0)
        self.assertEqual(payload["visualizations"], [])
        self.assertIsInstance(payload["signals"], list)
        self.assertTrue(any("no feasible visualization candidates" in s for s in payload["signals"]))

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame({"col_a": pd.Series([], dtype=float)})
        payload = build_visualization_payload(df, dataset_name="empty_df")

        self.assertEqual(payload["status"], "NO_MEANINGFUL_VISUALIZATION")
        self.assertEqual(payload["dataset_name"], "empty_df")
        self.assertEqual(payload["visualization_count"], 0)
        self.assertEqual(payload["visualizations"], [])
        self.assertTrue(any("dataframe has zero rows" in s for s in payload["signals"]))

    def test_one_row_dataframe(self) -> None:
        df = pd.DataFrame({"cat": ["A"], "val": [10.0]})
        payload = build_visualization_payload(df, dataset_name="one_row")

        self.assertEqual(payload["status"], "NO_MEANINGFUL_VISUALIZATION")
        self.assertEqual(payload["visualization_count"], 0)
        self.assertEqual(payload["visualizations"], [])
        self.assertTrue(
            any("insufficient rows for meaningful visualization (minimum 3 rows required)" in s for s in payload["signals"])
        )

    def test_two_row_dataframe(self) -> None:
        df = pd.DataFrame({"cat": ["A", "B"], "val": [10.0, 20.0]})
        payload = build_visualization_payload(df, dataset_name="two_rows")

        self.assertEqual(payload["status"], "NO_MEANINGFUL_VISUALIZATION")
        self.assertEqual(payload["visualization_count"], 0)
        self.assertEqual(payload["visualizations"], [])
        self.assertTrue(
            any("insufficient rows for meaningful visualization (minimum 3 rows required)" in s for s in payload["signals"])
        )


class TestJsonSerializability(unittest.TestCase):
    """Verify that all payloads are 100% JSON-serializable and round-trip without loss."""

    def test_ready_payload_json_serializable(self) -> None:
        df = _make_realistic_df()
        payload = build_visualization_payload(df, dataset_name="sales")

        # Must pass strict JSON type inspection
        _assert_json_safe_types_only(payload)

        # Must dump and reload without error
        serialized = json.dumps(payload)
        self.assertIsInstance(serialized, str)
        reloaded = json.loads(serialized)
        self.assertEqual(reloaded, payload)

    def test_no_meaningful_payload_json_serializable(self) -> None:
        df = pd.DataFrame({"x": [1, 2]})
        payload = build_visualization_payload(df)

        _assert_json_safe_types_only(payload)

        serialized = json.dumps(payload)
        self.assertIsInstance(serialized, str)
        reloaded = json.loads(serialized)
        self.assertEqual(reloaded, payload)


class TestPreservationOfStage5Attributes(unittest.TestCase):
    """Verify that Stage 6 strictly preserves Stage 5 ordering, scores, chart data, and signals."""

    def test_preservation_of_ordering(self) -> None:
        df = _make_realistic_df()
        stage5_result = build_visualization_result(df, max_visualizations=5)
        payload = build_visualization_payload(df, max_visualizations=5)

        self.assertEqual(len(payload["visualizations"]), stage5_result.visualization_count)

        payload_ranks = [v["rank"] for v in payload["visualizations"]]
        stage5_ranks = [v.rank for v in stage5_result.visualizations]
        self.assertEqual(payload_ranks, stage5_ranks)

        payload_chart_types = [v["chart_type"] for v in payload["visualizations"]]
        stage5_chart_types = [v.chart_type for v in stage5_result.visualizations]
        self.assertEqual(payload_chart_types, stage5_chart_types)

        payload_x_cols = [v["x_column"] for v in payload["visualizations"]]
        stage5_x_cols = [v.x_column for v in stage5_result.visualizations]
        self.assertEqual(payload_x_cols, stage5_x_cols)

        payload_y_cols = [v["y_column"] for v in payload["visualizations"]]
        stage5_y_cols = [v.y_column for v in stage5_result.visualizations]
        self.assertEqual(payload_y_cols, stage5_y_cols)

    def test_preservation_of_scores(self) -> None:
        df = _make_realistic_df()
        stage5_result = build_visualization_result(df, max_visualizations=5)
        payload = build_visualization_payload(df, max_visualizations=5)

        payload_scores = [v["score"] for v in payload["visualizations"]]
        stage5_scores = [v.score for v in stage5_result.visualizations]
        self.assertEqual(payload_scores, stage5_scores)

    def test_preservation_of_chart_data(self) -> None:
        df = _make_realistic_df()
        stage5_result = build_visualization_result(df, max_visualizations=5)
        payload = build_visualization_payload(df, max_visualizations=5)

        for i, item in enumerate(stage5_result.visualizations):
            expected_cd = item.chart_data.model_dump(mode="json")
            actual_cd = payload["visualizations"][i]["chart_data"]
            self.assertEqual(actual_cd, expected_cd)

    def test_preservation_of_signals(self) -> None:
        df = _make_realistic_df()
        stage5_result = build_visualization_result(df, max_visualizations=5)
        payload = build_visualization_payload(df, max_visualizations=5)

        # Top-level signals
        self.assertEqual(payload["signals"], stage5_result.signals)

        # Item-level signals
        for i, item in enumerate(stage5_result.visualizations):
            self.assertEqual(payload["visualizations"][i]["signals"], item.signals)


class TestInputValidation(unittest.TestCase):
    """Verify input validation for dataframe and max_visualizations."""

    def test_non_dataframe_raises_typeerror(self) -> None:
        with self.assertRaises(TypeError):
            build_visualization_payload([1, 2, 3])  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            build_visualization_payload("not a df")  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            build_visualization_payload(None)  # type: ignore[arg-type]

    def test_invalid_max_visualizations_raises_valueerror(self) -> None:
        df = _make_realistic_df()

        with self.assertRaises(ValueError):
            build_visualization_payload(df, max_visualizations=0)

        with self.assertRaises(ValueError):
            build_visualization_payload(df, max_visualizations=-3)

        with self.assertRaises(ValueError):
            build_visualization_payload(df, max_visualizations="5")  # type: ignore[arg-type]

    def test_bool_max_visualizations_rejected(self) -> None:
        df = _make_realistic_df()

        with self.assertRaises(ValueError):
            build_visualization_payload(df, max_visualizations=True)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            build_visualization_payload(df, max_visualizations=False)  # type: ignore[arg-type]


class TestDeterministicExecution(unittest.TestCase):
    """Verify deterministic repeatability."""

    def test_repeated_execution_yields_identical_payloads(self) -> None:
        df = _make_realistic_df()
        p1 = build_visualization_payload(df, dataset_name="sales", max_visualizations=4)
        p2 = build_visualization_payload(df, dataset_name="sales", max_visualizations=4)

        self.assertEqual(p1, p2)
        self.assertEqual(json.dumps(p1, sort_keys=True), json.dumps(p2, sort_keys=True))

    def test_dataset_name_handling(self) -> None:
        df = _make_realistic_df()
        p_named = build_visualization_payload(df, dataset_name="my_ds")
        self.assertEqual(p_named["dataset_name"], "my_ds")

        p_unnamed = build_visualization_payload(df, dataset_name=None)
        self.assertIsNone(p_unnamed["dataset_name"])


if __name__ == "__main__":
    unittest.main()

