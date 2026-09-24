"""Tests for Visualization Request Parser — Pipeline 3, Stage 7.

Validates deterministic extraction of chart types, aggregations, intents,
and columns from natural language visualization requests.
"""

from __future__ import annotations

import unittest
import pandas as pd

from pipelines.pipeline3_results.visualization.column_classifier import classify_columns
from pipelines.pipeline3_results.visualization.request_parser import parse_visualization_request


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
        "pizza_ingredients": ["Ham, Cheese", "Cheese, Sauce", "Peppers, Onions", "BBQ, Chicken", "Pepperoni"],
        "pizza_name": ["The Hawaiian Pizza", "The Classic Deluxe", "The Veggie Pizza", "The BBQ Pizza", "Pepperoni"],
    })


class TestChartTypeExtraction(unittest.TestCase):
    """Test recognition of all 5 supported chart types."""

    def setUp(self) -> None:
        self.df = _sample_pizza_df()

    def test_explicit_bar_chart(self) -> None:
        req = parse_visualization_request("Give me a bar chart of total_price by pizza_category", self.df)
        self.assertEqual(req.requested_chart_type, "BAR")
        self.assertTrue(req.is_explicit_chart_type)
        self.assertTrue(any("explicit chart type detected" in s and "BAR" in s for s in req.parser_signals))

    def test_explicit_line_chart(self) -> None:
        req = parse_visualization_request("Plot a line graph of total_price over order_date", self.df)
        self.assertEqual(req.requested_chart_type, "LINE")
        self.assertTrue(req.is_explicit_chart_type)
        self.assertTrue(any("explicit chart type detected" in s and "LINE" in s for s in req.parser_signals))

    def test_explicit_scatter_plot(self) -> None:
        req = parse_visualization_request("Make a scatter plot of unit_price and total_price", self.df)
        self.assertEqual(req.requested_chart_type, "SCATTER")
        self.assertTrue(req.is_explicit_chart_type)
        self.assertTrue(any("explicit chart type detected" in s and "SCATTER" in s for s in req.parser_signals))

    def test_explicit_histogram(self) -> None:
        req = parse_visualization_request("Show a histogram of total_price", self.df)
        self.assertEqual(req.requested_chart_type, "HISTOGRAM")
        self.assertTrue(req.is_explicit_chart_type)
        self.assertTrue(any("explicit chart type detected" in s and "HISTOGRAM" in s for s in req.parser_signals))

    def test_explicit_count_bar(self) -> None:
        req = parse_visualization_request("Show a count bar chart of pizza_size", self.df)
        self.assertEqual(req.requested_chart_type, "COUNT_BAR")
        self.assertTrue(req.is_explicit_chart_type)


class TestAggregationExtraction(unittest.TestCase):
    """Test recognition of supported aggregation methods."""

    def setUp(self) -> None:
        self.df = _sample_pizza_df()

    def test_count_aggregation(self) -> None:
        req = parse_visualization_request("Show the count of pizza_category", self.df)
        self.assertEqual(req.aggregation, "COUNT")

    def test_sum_aggregation(self) -> None:
        req = parse_visualization_request("Show the total sales by pizza_category", self.df)
        self.assertEqual(req.aggregation, "SUM")

    def test_mean_aggregation(self) -> None:
        req = parse_visualization_request("Show average total_price by pizza_size", self.df)
        self.assertEqual(req.aggregation, "MEAN")

    def test_median_aggregation(self) -> None:
        req = parse_visualization_request("Show median total_price by pizza_category", self.df)
        self.assertEqual(req.aggregation, "MEDIAN")

    def test_min_aggregation(self) -> None:
        req = parse_visualization_request("Show minimum unit_price by pizza_category", self.df)
        self.assertEqual(req.aggregation, "MIN")

    def test_max_aggregation(self) -> None:
        req = parse_visualization_request("Show maximum total_price by pizza_size", self.df)
        self.assertEqual(req.aggregation, "MAX")


class TestColumnMatching(unittest.TestCase):
    """Test deterministic column name resolution across various string formats."""

    def setUp(self) -> None:
        self.df = _sample_pizza_df()

    def test_exact_column_match(self) -> None:
        req = parse_visualization_request("Show total_price by pizza_category", self.df)
        self.assertIn("pizza_category", req.requested_columns)
        self.assertIn("total_price", req.requested_columns)

    def test_case_insensitive_match(self) -> None:
        req = parse_visualization_request("Show Total_Price by Pizza_Category", self.df)
        self.assertIn("pizza_category", req.requested_columns)
        self.assertIn("total_price", req.requested_columns)

    def test_spaces_vs_underscores(self) -> None:
        req = parse_visualization_request("Show total price by pizza category", self.df)
        self.assertIn("pizza_category", req.requested_columns)
        self.assertIn("total_price", req.requested_columns)

    def test_hyphen_vs_underscore(self) -> None:
        req = parse_visualization_request("Show total-price by pizza-category", self.df)
        self.assertIn("pizza_category", req.requested_columns)
        self.assertIn("total_price", req.requested_columns)

    def test_token_submatch(self) -> None:
        # 'category' matches pizza_category because it is a unique token in columns
        req = parse_visualization_request("Show sales by category", self.df)
        self.assertIn("pizza_category", req.requested_columns)

    def test_semantic_alias_sales_matches_total_price(self) -> None:
        req = parse_visualization_request("Show sales by pizza_category", self.df)
        self.assertIn("total_price", req.requested_columns)
        self.assertTrue(any("semantic_alias" in s for s in req.parser_signals))


class TestPrepositionResolution(unittest.TestCase):
    """Test X/Y column resolution using grammatical prepositions."""

    def setUp(self) -> None:
        self.df = _sample_pizza_df()

    def test_y_by_x_resolution(self) -> None:
        req = parse_visualization_request("Show total_price by pizza_category", self.df)
        self.assertEqual(req.x_column, "pizza_category")
        self.assertEqual(req.y_column, "total_price")

    def test_y_over_x_resolution(self) -> None:
        req = parse_visualization_request("Plot total_price over order_date", self.df)
        self.assertEqual(req.x_column, "order_date")
        self.assertEqual(req.y_column, "total_price")

    def test_x_versus_y_resolution(self) -> None:
        req = parse_visualization_request("Show unit_price versus total_price", self.df)
        self.assertEqual(req.x_column, "unit_price")
        self.assertEqual(req.y_column, "total_price")


class TestExplicitVsInferredDistinction(unittest.TestCase):
    """Verify that the parser distinguishes explicit chart types from inferred types."""

    def setUp(self) -> None:
        self.df = _sample_pizza_df()

    def test_explicit_chart_type_flag(self) -> None:
        req = parse_visualization_request("Make a bar chart of total_price by pizza_size", self.df)
        self.assertEqual(req.requested_chart_type, "BAR")
        self.assertTrue(req.is_explicit_chart_type)

    def test_inferred_chart_type_flag(self) -> None:
        req = parse_visualization_request("Show total_price by pizza_size", self.df)
        self.assertEqual(req.requested_chart_type, "BAR")
        self.assertFalse(req.is_explicit_chart_type)
        self.assertTrue(any("chart type inferred from" in s for s in req.parser_signals))


class TestEdgeCasesAndValidation(unittest.TestCase):
    """Test edge cases, empty input, and type checks."""

    def setUp(self) -> None:
        self.df = _sample_pizza_df()

    def test_empty_string_request(self) -> None:
        req = parse_visualization_request("", self.df)
        self.assertIsNone(req.requested_chart_type)
        self.assertIsNone(req.x_column)
        self.assertIsNone(req.y_column)
        self.assertEqual(req.requested_columns, [])
        self.assertTrue(any("empty request string" in s for s in req.parser_signals))

    def test_whitespace_only_request(self) -> None:
        req = parse_visualization_request("    ", self.df)
        self.assertIsNone(req.requested_chart_type)
        self.assertTrue(any("empty request string" in s for s in req.parser_signals))

    def test_non_string_raises_typeerror(self) -> None:
        with self.assertRaises(TypeError):
            parse_visualization_request(123, self.df)  # type: ignore[arg-type]

    def test_non_dataframe_raises_typeerror(self) -> None:
        with self.assertRaises(TypeError):
            parse_visualization_request("Show sales", [1, 2, 3])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

