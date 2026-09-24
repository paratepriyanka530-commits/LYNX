"""Tests for Visualization Request Validator — Pipeline 3, Stage 7.

Validates structural compatibility checks, rejection of invalid roles,
clarification states for ambiguous requests, determinism, and JSON serializability.
"""

from __future__ import annotations

import json
import unittest

import pandas as pd

from pipelines.pipeline3_results.visualization.column_classifier import classify_columns
from pipelines.pipeline3_results.visualization.request_parser import parse_visualization_request
from pipelines.pipeline3_results.visualization.request_validator import validate_visualization_request
from pipelines.pipeline3_results.visualization.schemas import VisualizationRequest


def _sample_pizza_df() -> pd.DataFrame:
    """Create a sample DataFrame matching the pizza_sales schema."""
    return pd.DataFrame({
        "pizza_id": [1, 2, 3, 4, 5],
        "order_id": [101, 101, 102, 102, 103],
        "pizza_name_id": ["hawaiian_m", "classic_l", "veggie_s", "bbq_m", "pep_l"],
        "quantity": [1, 2, 1, 3, 1],
        "order_date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03"],
        "order_time": ["11:30:00", "11:35:00", "12:15:00", "12:20:00", "13:00:00"],
        "unit_price": [12.50, 16.00, 10.00, 15.50, 18.00],
        "total_price": [12.50, 32.00, 10.00, 46.50, 18.00],
        "pizza_size": ["M", "L", "S", "M", "L"],
        "pizza_category": ["Classic", "Classic", "Veggie", "Supreme", "Classic"],
        "pizza_ingredients": [
            "Fresh sliced ham, sweet tropical pineapple, rich mozzarella cheese, and signature tomato sauce",
            "Grated parmesan, fresh whole-milk mozzarella cheese, garlic herb butter, and classic marinara",
            "Fire-roasted bell peppers, red onions, button mushrooms, baby spinach, and seasoned olive oil",
            "Smoked barbecue pulled chicken, red onions, tangy house BBQ sauce, and aged cheddar cheese",
            "Crispy spicy beef pepperoni slices, shredded mozzarella cheese, and seasoned Italian pizza sauce",
        ],
        "pizza_name": ["The Hawaiian Pizza", "The Classic Deluxe", "The Veggie Pizza", "The BBQ Pizza", "Pepperoni"],
    })


class TestValidRequests(unittest.TestCase):
    """Test validation of valid visualization requests across all chart types."""

    def setUp(self) -> None:
        self.df = _sample_pizza_df()
        self.classification = classify_columns(self.df)

    def test_valid_bar_chart(self) -> None:
        req = parse_visualization_request("Show total_price by pizza_category", self.df, self.classification)
        res = validate_visualization_request(self.df, self.classification, req)

        self.assertEqual(res.status, "VALID")
        self.assertEqual(res.chart_type, "BAR")
        self.assertEqual(res.x_column, "pizza_category")
        self.assertEqual(res.y_column, "total_price")
        self.assertEqual(res.errors, [])
        self.assertTrue(any("BAR chart validated" in s for s in res.signals))

    def test_valid_count_bar(self) -> None:
        req = parse_visualization_request("Show count of pizza_size", self.df, self.classification)
        res = validate_visualization_request(self.df, self.classification, req)

        self.assertEqual(res.status, "VALID")
        self.assertEqual(res.chart_type, "COUNT_BAR")
        self.assertEqual(res.x_column, "pizza_size")
        self.assertIsNone(res.y_column)
        self.assertEqual(res.errors, [])

    def test_valid_line_chart(self) -> None:
        req = parse_visualization_request("Plot total_price over order_date", self.df, self.classification)
        res = validate_visualization_request(self.df, self.classification, req)

        self.assertEqual(res.status, "VALID")
        self.assertEqual(res.chart_type, "LINE")
        self.assertEqual(res.x_column, "order_date")
        self.assertEqual(res.y_column, "total_price")
        self.assertEqual(res.errors, [])

    def test_valid_scatter_plot(self) -> None:
        req = parse_visualization_request("Scatter plot of unit_price and total_price", self.df, self.classification)
        res = validate_visualization_request(self.df, self.classification, req)

        self.assertEqual(res.status, "VALID")
        self.assertEqual(res.chart_type, "SCATTER")
        self.assertEqual(res.x_column, "unit_price")
        self.assertEqual(res.y_column, "total_price")
        self.assertEqual(res.errors, [])

    def test_valid_histogram(self) -> None:
        req = parse_visualization_request("Histogram of unit_price", self.df, self.classification)
        res = validate_visualization_request(self.df, self.classification, req)

        self.assertEqual(res.status, "VALID")
        self.assertEqual(res.chart_type, "HISTOGRAM")
        self.assertEqual(res.x_column, "unit_price")
        self.assertIsNone(res.y_column)
        self.assertEqual(res.errors, [])


class TestInvalidRequests(unittest.TestCase):
    """Test proper rejection of invalid or structurally contradictory requests."""

    def setUp(self) -> None:
        self.df = _sample_pizza_df()
        self.classification = classify_columns(self.df)

    def test_identifier_column_rejected(self) -> None:
        req = parse_visualization_request("Histogram of pizza_id", self.df, self.classification)
        res = validate_visualization_request(self.df, self.classification, req)

        self.assertEqual(res.status, "INVALID")
        self.assertTrue(any("IDENTIFIER" in e for e in res.errors))

    def test_two_identifiers_rejected(self) -> None:
        req = parse_visualization_request("Plot order_id against pizza_id", self.df, self.classification)
        res = validate_visualization_request(self.df, self.classification, req)

        self.assertEqual(res.status, "INVALID")
        self.assertTrue(any("Both requested columns are IDENTIFIER fields" in e for e in res.errors))

    def test_descriptive_text_rejected(self) -> None:
        req = parse_visualization_request("Histogram of pizza_ingredients", self.df, self.classification)
        res = validate_visualization_request(self.df, self.classification, req)

        self.assertEqual(res.status, "INVALID")
        self.assertTrue(any("DESCRIPTIVE_TEXT" in e for e in res.errors))

    def test_nonexistent_column_rejected(self) -> None:
        # Create a manual request referencing a fake column
        req = VisualizationRequest(
            raw_request="Bar chart of unknown_col by pizza_category",
            requested_chart_type="BAR",
            x_column="pizza_category",
            y_column="unknown_col",
            requested_columns=["pizza_category", "unknown_col"],
        )
        res = validate_visualization_request(self.df, self.classification, req)

        self.assertEqual(res.status, "INVALID")
        self.assertTrue(any("does not exist" in e for e in res.errors))

    def test_histogram_on_category_rejected(self) -> None:
        req = parse_visualization_request("Histogram of pizza_category", self.df, self.classification)
        res = validate_visualization_request(self.df, self.classification, req)

        self.assertEqual(res.status, "INVALID")
        self.assertTrue(any("HISTOGRAM requires a MEASURE column" in e for e in res.errors))

    def test_count_bar_on_measure_rejected(self) -> None:
        req = parse_visualization_request("Count bar of total_price", self.df, self.classification)
        res = validate_visualization_request(self.df, self.classification, req)

        self.assertEqual(res.status, "INVALID")
        self.assertTrue(any("COUNT_BAR requires a CATEGORY or BOOLEAN column" in e for e in res.errors))

    def test_scatter_on_identical_columns_rejected(self) -> None:
        req = VisualizationRequest(
            raw_request="Scatter unit_price against unit_price",
            requested_chart_type="SCATTER",
            x_column="unit_price",
            y_column="unit_price",
            requested_columns=["unit_price", "unit_price"],
        )
        res = validate_visualization_request(self.df, self.classification, req)

        self.assertEqual(res.status, "INVALID")
        self.assertTrue(any("identical columns" in e for e in res.errors))

    def test_missing_temporal_column_rejects_line(self) -> None:
        # DataFrame without any temporal column
        df_no_temp = pd.DataFrame({"cat": ["A", "B", "C"], "val": [1.0, 2.0, 3.0]})
        clf_no_temp = classify_columns(df_no_temp)
        req = VisualizationRequest(
            raw_request="Plot val over time",
            requested_chart_type="LINE",
            x_column="time",
            y_column="val",
        )
        res = validate_visualization_request(df_no_temp, clf_no_temp, req)

        self.assertEqual(res.status, "INVALID")
        self.assertTrue(any("No suitable temporal column was found" in e for e in res.errors))

    def test_insufficient_measures_rejects_scatter(self) -> None:
        # DataFrame with only 1 measure
        df_one_meas = pd.DataFrame({"cat": ["A", "B", "C"], "val": [1.0, 2.0, 3.0]})
        clf_one_meas = classify_columns(df_one_meas)
        req = VisualizationRequest(
            raw_request="Make a scatter plot",
            requested_chart_type="SCATTER",
        )
        res = validate_visualization_request(df_one_meas, clf_one_meas, req)

        self.assertEqual(res.status, "INVALID")
        self.assertTrue(any("At least two MEASURE columns are required" in e for e in res.errors))


class TestClarificationRequests(unittest.TestCase):
    """Test structured ambiguity handling (NEEDS_CLARIFICATION)."""

    def setUp(self) -> None:
        self.df = _sample_pizza_df()
        self.classification = classify_columns(self.df)

    def test_generic_graph_request(self) -> None:
        req = parse_visualization_request("Make me a graph", self.df, self.classification)
        res = validate_visualization_request(self.df, self.classification, req)

        self.assertEqual(res.status, "NEEDS_CLARIFICATION")
        self.assertEqual(res.errors, [])
        self.assertGreater(len(res.clarification_options), 0)
        self.assertTrue(any("no chart type specified" in s for s in res.signals))

    def test_scatter_without_columns_clarification(self) -> None:
        req = parse_visualization_request("Make a scatter plot", self.df, self.classification)
        res = validate_visualization_request(self.df, self.classification, req)

        self.assertEqual(res.status, "NEEDS_CLARIFICATION")
        self.assertEqual(res.chart_type, "SCATTER")
        self.assertIn("unit_price", res.clarification_options)
        self.assertIn("total_price", res.clarification_options)

    def test_line_without_measure_clarification(self) -> None:
        req = parse_visualization_request("Line chart", self.df, self.classification)
        res = validate_visualization_request(self.df, self.classification, req)

        self.assertEqual(res.status, "NEEDS_CLARIFICATION")
        self.assertEqual(res.chart_type, "LINE")
        self.assertEqual(res.x_column, "order_date")
        self.assertIn("total_price", res.clarification_options)


class TestDeterminismAndJsonSerialization(unittest.TestCase):
    """Verify deterministic repeatability and JSON serializability."""

    def setUp(self) -> None:
        self.df = _sample_pizza_df()
        self.classification = classify_columns(self.df)

    def test_deterministic_output(self) -> None:
        req1 = parse_visualization_request("Show total_price by pizza_category", self.df, self.classification)
        res1 = validate_visualization_request(self.df, self.classification, req1)

        req2 = parse_visualization_request("Show total_price by pizza_category", self.df, self.classification)
        res2 = validate_visualization_request(self.df, self.classification, req2)

        self.assertEqual(res1.to_dict(), res2.to_dict())

    def test_json_serializability(self) -> None:
        req = parse_visualization_request("Show total_price by pizza_category", self.df, self.classification)
        res = validate_visualization_request(self.df, self.classification, req)

        d = res.to_dict()
        serialized = json.dumps(d)
        reloaded = json.loads(serialized)
        self.assertEqual(reloaded, d)

    def test_type_guards(self) -> None:
        req = parse_visualization_request("Show total_price by pizza_category", self.df, self.classification)

        with self.assertRaises(TypeError):
            validate_visualization_request("not a df", self.classification, req)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            validate_visualization_request(self.df, "not a classification", req)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            validate_visualization_request(self.df, self.classification, "not a request")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
