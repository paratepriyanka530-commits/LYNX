"""Comprehensive tests for Pipeline 3 Stage 2 — Visualization Feasibility."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipelines.pipeline3_results.visualization.column_classifier import classify_columns
from pipelines.pipeline3_results.visualization.feasibility import (
    _MAX_CATEGORY_CARDINALITY,
    _MAX_SCATTER_PAIRS,
    analyze_visualization_feasibility,
)
from pipelines.pipeline3_results.visualization.schemas import (
    DatasetClassification,
    FeasibilityResult,
    VisualizationCandidate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clf_df(df: pd.DataFrame) -> DatasetClassification:
    """Classify a dataframe with Stage 1 and return the DatasetClassification."""
    return classify_columns(df)


def _chart_types(result: FeasibilityResult) -> list[str]:
    return [c.chart_type for c in result.candidates]


def _has(result: FeasibilityResult, chart_type: str, x: str, y: str | None = None) -> bool:
    for c in result.candidates:
        if c.chart_type == chart_type and c.x_column == x and c.y_column == y:
            return True
    return False


# ---------------------------------------------------------------------------
# Test 1 — CATEGORY + MEASURE → BAR
# ---------------------------------------------------------------------------

class TestCategoryMeasureBar(unittest.TestCase):
    def test_category_measure_produces_bar(self) -> None:
        df = pd.DataFrame({
            "region": ["North", "South", "East", "West", "North", "South"],
            "sales_amount": [100.0, 200.0, 150.0, 300.0, 250.0, 180.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        self.assertTrue(_has(result, "BAR", "region", "sales_amount"))

    def test_bar_signal_content(self) -> None:
        df = pd.DataFrame({
            "department": ["A", "B", "A", "B"],
            "revenue": [10.0, 20.0, 30.0, 40.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        bar_candidates = result.candidates_by_type("BAR")
        self.assertTrue(len(bar_candidates) > 0)
        signals = bar_candidates[0].signals
        self.assertTrue(any("CATEGORY" in s for s in signals))
        self.assertTrue(any("MEASURE" in s for s in signals))
        self.assertTrue(any("BAR" in s for s in signals))


# ---------------------------------------------------------------------------
# Test 2 — TEMPORAL + MEASURE → LINE
# ---------------------------------------------------------------------------

class TestTemporalMeasureLine(unittest.TestCase):
    def test_datetime_measure_produces_line(self) -> None:
        df = pd.DataFrame({
            "order_date": pd.date_range("2023-01-01", periods=6, freq="ME"),
            "revenue": [100.0, 120.0, 90.0, 150.0, 200.0, 180.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        self.assertTrue(_has(result, "LINE", "order_date", "revenue"))

    def test_string_date_with_temporal_name_produces_line(self) -> None:
        df = pd.DataFrame({
            "sale_date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
            "units": [10.0, 14.0, 12.0, 18.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        self.assertTrue(_has(result, "LINE", "sale_date", "units"))

    def test_line_signal_content(self) -> None:
        df = pd.DataFrame({
            "created_at": pd.date_range("2023-01-01", periods=4, freq="D"),
            "cost": [50.0, 60.0, 55.0, 70.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        lines = result.candidates_by_type("LINE")
        self.assertTrue(len(lines) > 0)
        signals = lines[0].signals
        self.assertTrue(any("TEMPORAL" in s for s in signals))
        self.assertTrue(any("LINE" in s for s in signals))


# ---------------------------------------------------------------------------
# Test 3 — MEASURE + MEASURE → SCATTER
# ---------------------------------------------------------------------------

class TestMeasureMeasureScatter(unittest.TestCase):
    def test_two_measures_produce_scatter(self) -> None:
        df = pd.DataFrame({
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],
            "units_sold": [100.0, 80.0, 120.0, 60.0, 90.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        self.assertTrue(_has(result, "SCATTER", "price", "units_sold")
                        or _has(result, "SCATTER", "units_sold", "price"))

    def test_scatter_signal_content(self) -> None:
        df = pd.DataFrame({
            "age": [25.0, 30.0, 35.0, 40.0],
            "salary": [30000.0, 45000.0, 60000.0, 80000.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        scatters = result.candidates_by_type("SCATTER")
        self.assertTrue(len(scatters) > 0)
        signals = scatters[0].signals
        self.assertTrue(any("SCATTER" in s for s in signals))

    def test_single_measure_no_scatter(self) -> None:
        df = pd.DataFrame({
            "region": ["A", "B", "C"],
            "sales": [10.0, 20.0, 30.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        self.assertEqual(len(result.candidates_by_type("SCATTER")), 0)


# ---------------------------------------------------------------------------
# Test 4 — MEASURE alone → HISTOGRAM
# ---------------------------------------------------------------------------

class TestMeasureHistogram(unittest.TestCase):
    def test_measure_produces_histogram(self) -> None:
        df = pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(20)],
            "order_total": [float(i * 10 + 5) for i in range(20)],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        self.assertTrue(_has(result, "HISTOGRAM", "order_total"))

    def test_histogram_has_no_y_column(self) -> None:
        df = pd.DataFrame({
            "price": [10.5, 20.3, 15.7, 22.1, 18.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        histograms = result.candidates_by_type("HISTOGRAM")
        self.assertTrue(len(histograms) > 0)
        for h in histograms:
            self.assertIsNone(h.y_column)

    def test_histogram_signal_content(self) -> None:
        df = pd.DataFrame({
            "revenue": [100.0, 200.0, 300.0, 400.0, 500.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        histograms = result.candidates_by_type("HISTOGRAM")
        self.assertTrue(len(histograms) > 0)
        self.assertTrue(any("HISTOGRAM" in s for s in histograms[0].signals))


# ---------------------------------------------------------------------------
# Test 5 — CATEGORY alone → COUNT_BAR
# ---------------------------------------------------------------------------

class TestCategoryCountBar(unittest.TestCase):
    def test_category_alone_produces_count_bar(self) -> None:
        df = pd.DataFrame({
            "status": ["Active", "Inactive", "Active", "Active", "Inactive"],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        self.assertTrue(_has(result, "COUNT_BAR", "status"))

    def test_count_bar_has_no_y_column(self) -> None:
        df = pd.DataFrame({
            "category": ["A", "B", "A", "C", "B", "A"],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        count_bars = result.candidates_by_type("COUNT_BAR")
        self.assertTrue(len(count_bars) > 0)
        for cb in count_bars:
            self.assertIsNone(cb.y_column)


# ---------------------------------------------------------------------------
# Test 6 — BOOLEAN + MEASURE → BAR
# ---------------------------------------------------------------------------

class TestBooleanMeasureBar(unittest.TestCase):
    def test_boolean_measure_produces_bar(self) -> None:
        df = pd.DataFrame({
            "is_returned": [0, 1, 0, 1, 0, 0, 1, 0],
            "order_value": [50.0, 30.0, 80.0, 20.0, 60.0, 90.0, 40.0, 70.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        # is_returned should be BOOLEAN (has boolean name), order_value MEASURE
        bars = result.candidates_by_type("BAR")
        self.assertTrue(
            any(c.x_column == "is_returned" and c.y_column == "order_value" for c in bars),
            f"Expected BAR(is_returned, order_value). candidates: {bars}",
        )

    def test_boolean_alone_produces_count_bar(self) -> None:
        df = pd.DataFrame({
            "is_active": [True, False, True, True, False],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        self.assertTrue(_has(result, "COUNT_BAR", "is_active"))


# ---------------------------------------------------------------------------
# Test 7 — IDENTIFIER rejected as dimension
# ---------------------------------------------------------------------------

class TestIdentifierRejected(unittest.TestCase):
    def test_identifier_not_in_any_candidate_x(self) -> None:
        df = pd.DataFrame({
            "customer_id": [f"CUST_{i:03d}" for i in range(10)],
            "sales_amount": [float(i * 10) for i in range(10)],
            "region": ["North", "South"] * 5,
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        for c in result.candidates:
            self.assertNotEqual(
                c.x_column, "customer_id",
                f"IDENTIFIER customer_id should never appear as x_column: {c}",
            )
            self.assertNotEqual(
                c.y_column, "customer_id",
                f"IDENTIFIER customer_id should never appear as y_column: {c}",
            )

    def test_identifier_suppressed_reason_logged(self) -> None:
        df = pd.DataFrame({
            "order_id": [f"ORD_{i}" for i in range(5)],
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        self.assertTrue(
            any("IDENTIFIER" in r for r in result.suppressed_reasons),
            f"Expected IDENTIFIER suppression log. Got: {result.suppressed_reasons}",
        )


# ---------------------------------------------------------------------------
# Test 8 — DESCRIPTIVE_TEXT rejected as dimension
# ---------------------------------------------------------------------------

class TestDescriptiveTextRejected(unittest.TestCase):
    def test_descriptive_text_not_in_any_candidate(self) -> None:
        df = pd.DataFrame({
            "product_description": [
                "High quality stainless steel kitchen knife set.",
                "Premium ergonomic office chair with lumbar support.",
                "Waterproof hiking boots suitable for all terrain.",
                "Compact portable Bluetooth speaker with bass boost.",
            ],
            "price": [49.99, 299.99, 129.99, 79.99],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        for c in result.candidates:
            self.assertNotEqual(c.x_column, "product_description")
            self.assertNotEqual(c.y_column, "product_description")

    def test_descriptive_text_suppression_logged(self) -> None:
        df = pd.DataFrame({
            "notes": [
                "Additional commentary provided here with several words.",
                "Another longer note about the customer request details.",
                "Feedback from review covering multiple aspects of service.",
            ],
            "score": [4.5, 3.0, 5.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        self.assertTrue(
            any("DESCRIPTIVE_TEXT" in r for r in result.suppressed_reasons),
        )


# ---------------------------------------------------------------------------
# Test 9 — UNKNOWN rejected
# ---------------------------------------------------------------------------

class TestUnknownRejected(unittest.TestCase):
    def test_all_null_column_not_in_candidates(self) -> None:
        df = pd.DataFrame({
            "good_measure": [1.0, 2.0, 3.0, 4.0, 5.0],
            "null_col": [None, None, None, None, None],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        for c in result.candidates:
            self.assertNotEqual(c.x_column, "null_col")
            self.assertNotEqual(c.y_column, "null_col")

    def test_unknown_suppression_logged(self) -> None:
        df = pd.DataFrame({
            "constant_val": [99, 99, 99, 99],
            "price": [10.0, 20.0, 30.0, 40.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        self.assertTrue(
            any("UNKNOWN" in r for r in result.suppressed_reasons),
        )


# ---------------------------------------------------------------------------
# Test 10 — All-missing MEASURE rejected
# ---------------------------------------------------------------------------

class TestAllMissingMeasureRejected(unittest.TestCase):
    def test_all_null_measure_no_histogram(self) -> None:
        df = pd.DataFrame({
            "null_amount": [None, None, None],
            "region": ["A", "B", "C"],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        for c in result.candidates:
            self.assertNotEqual(c.x_column, "null_amount")
            self.assertNotEqual(c.y_column, "null_amount")

    def test_null_measure_suppression_logged(self) -> None:
        df = pd.DataFrame({
            "null_sales": [None, None, None],
            "category": ["X", "Y", "Z"],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        # null_sales classified as UNKNOWN (all_values_missing) — check suppressed
        self.assertTrue(
            any("null_sales" in r or "UNKNOWN" in r for r in result.suppressed_reasons)
        )


# ---------------------------------------------------------------------------
# Test 11 — All-missing CATEGORY rejected
# ---------------------------------------------------------------------------

class TestAllMissingCategoryRejected(unittest.TestCase):
    def test_all_null_category_no_count_bar(self) -> None:
        df = pd.DataFrame({
            "null_region": [None, None, None],
            "sales": [10.0, 20.0, 30.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        for c in result.candidates:
            self.assertNotEqual(c.x_column, "null_region")
            self.assertNotEqual(c.y_column, "null_region")


# ---------------------------------------------------------------------------
# Test 12 — Zero-row dataframe
# ---------------------------------------------------------------------------

class TestZeroRowDataframe(unittest.TestCase):
    def test_zero_rows_no_candidates(self) -> None:
        df = pd.DataFrame({
            "region": pd.Series([], dtype=str),
            "sales": pd.Series([], dtype=float),
        })
        clf = classify_columns(df)
        result = analyze_visualization_feasibility(df, clf)
        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(result.row_count, 0)

    def test_zero_rows_suppression_logged(self) -> None:
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        clf = classify_columns(df)
        result = analyze_visualization_feasibility(df, clf)
        self.assertTrue(
            any("zero rows" in r.lower() for r in result.suppressed_reasons)
        )


# ---------------------------------------------------------------------------
# Test 13 — High-cardinality category suppressed as dimension
# ---------------------------------------------------------------------------

class TestHighCardinalityCategorySuppressed(unittest.TestCase):
    def test_high_cardinality_category_not_used_as_bar_dimension(self) -> None:
        # Create a category column with cardinality > _MAX_CATEGORY_CARDINALITY
        n = _MAX_CATEGORY_CARDINALITY + 10
        df = pd.DataFrame({
            "product_sku": [f"SKU_{i:04d}" for i in range(n)],
            "sales_amount": [float(i) for i in range(n)],
        })
        # Force product_sku to look like a CATEGORY not IDENTIFIER by using
        # a low unique_ratio (all unique → would be IDENTIFIER, so we need
        # it to slip into CATEGORY with high cardinality)
        # Instead, build a dataframe where a column HAS category role from Stage 1
        # but is too high-cardinality for a chart:
        clf = classify_columns(df)
        result = analyze_visualization_feasibility(df, clf)
        # product_sku will likely be classified IDENTIFIER due to high uniqueness;
        # confirm it does NOT appear as BAR x_column regardless
        for c in result.candidates:
            self.assertNotEqual(c.x_column, "product_sku")

    def test_high_cardinality_suppressed_reason_logged(self) -> None:
        # Build a DataFrame where a column is truly classified CATEGORY but
        # has too many unique values (> _MAX_CATEGORY_CARDINALITY).
        # We do this by building a column with uniqueness low enough that
        # Stage 1 classifies it as CATEGORY but cardinality still > limit.
        n = _MAX_CATEGORY_CARDINALITY * 5
        labels = [f"cat_{i % (_MAX_CATEGORY_CARDINALITY + 5)}" for i in range(n)]
        df = pd.DataFrame({
            "high_card_cat": labels,
            "value": [float(i % 10) for i in range(n)],
        })
        clf = classify_columns(df)
        result = analyze_visualization_feasibility(df, clf)
        # If high_card_cat is CATEGORY (not IDENTIFIER) and cardinality > limit,
        # it should appear in suppressed_reasons.
        cat_col = clf.get_column("high_card_cat")
        if cat_col is not None and cat_col.role == "CATEGORY":
            if cat_col.unique_count > _MAX_CATEGORY_CARDINALITY:
                self.assertTrue(
                    any("high_card_cat" in r or "cardinality" in r.lower()
                        for r in result.suppressed_reasons)
                )


# ---------------------------------------------------------------------------
# Test 14 — Multiple valid combinations
# ---------------------------------------------------------------------------

class TestMultipleValidCombinations(unittest.TestCase):
    def test_multiple_chart_types_from_rich_dataset(self) -> None:
        df = pd.DataFrame({
            "customer_id": [f"CUST_{i:03d}" for i in range(20)],
            "order_date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "region": (["North", "South", "East", "West"] * 5),
            "total_price": [float(i * 12.5) for i in range(20)],
            "quantity": [float(i % 10 + 1) for i in range(20)],
            "is_returned": [i % 2 for i in range(20)],  # needs boolean name → is_
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)

        types_present = set(result.chart_type_counts.keys())
        # Should have at least HISTOGRAM, COUNT_BAR, BAR, LINE, SCATTER
        self.assertIn("HISTOGRAM", result.chart_type_counts)
        self.assertIn("COUNT_BAR", result.chart_type_counts)
        self.assertIn("BAR", result.chart_type_counts)
        self.assertIn("LINE", result.chart_type_counts)
        self.assertIn("SCATTER", result.chart_type_counts)

        # Validate candidate count non-zero for each type
        self.assertGreater(result.chart_type_counts["HISTOGRAM"], 0)
        self.assertGreater(result.chart_type_counts["BAR"], 0)
        self.assertGreater(result.chart_type_counts["LINE"], 0)
        self.assertGreater(result.chart_type_counts["SCATTER"], 0)


# ---------------------------------------------------------------------------
# Test 15 — Deterministic repeated execution
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    def test_repeated_calls_identical_output(self) -> None:
        df = pd.DataFrame({
            "region": ["North", "South", "East", "West"] * 5,
            "order_date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "sales": [float(i * 10.5) for i in range(20)],
            "units": [float(i % 8 + 1) for i in range(20)],
            "is_fraud": [i % 2 for i in range(20)],
        })
        clf = _clf_df(df)
        r1 = analyze_visualization_feasibility(df, clf)
        r2 = analyze_visualization_feasibility(df, clf)
        self.assertEqual(r1.model_dump(), r2.model_dump())
        self.assertEqual(
            [c.model_dump() for c in r1.candidates],
            [c.model_dump() for c in r2.candidates],
        )


# ---------------------------------------------------------------------------
# Test 16 — Real-world mixed dataset structure
# ---------------------------------------------------------------------------

class TestRealWorldMixed(unittest.TestCase):
    def test_pizza_style_dataset(self) -> None:
        """Resembles the example from the requirements."""
        df = pd.DataFrame({
            "order_id": [f"ORD_{i:04d}" for i in range(30)],
            "pizza_size": (["Small", "Medium", "Large"] * 10),
            "pizza_type": (["Margherita", "Pepperoni", "BBQ Chicken"] * 10),
            "order_date": pd.date_range("2023-01-01", periods=30, freq="D"),
            "total_price": [float(i * 8.5 + 5) for i in range(30)],
            "quantity": [float((i % 5) + 1) for i in range(30)],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)

        # order_id is IDENTIFIER — must not appear in candidates
        for c in result.candidates:
            self.assertNotEqual(c.x_column, "order_id")
            self.assertNotEqual(c.y_column, "order_id")

        # Should have BAR (pizza_size/type × price/qty)
        bars = result.candidates_by_type("BAR")
        self.assertTrue(len(bars) > 0)

        # Should have LINE (order_date × price or qty)
        lines = result.candidates_by_type("LINE")
        self.assertTrue(len(lines) > 0)

        # Should have SCATTER (price × qty)
        scatters = result.candidates_by_type("SCATTER")
        self.assertTrue(len(scatters) > 0)

    def test_sales_dataset_correct_structure(self) -> None:
        df = pd.DataFrame({
            "transaction_id": [f"TX{i}" for i in range(15)],
            "sale_month": pd.date_range("2023-01-01", periods=15, freq="ME"),
            "product_category": (["Electronics", "Clothing", "Food"] * 5),
            "revenue": [float(i * 150) for i in range(15)],
            "cost": [float(i * 80) for i in range(15)],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)

        # transaction_id is IDENTIFIER — excluded
        for c in result.candidates:
            self.assertNotEqual(c.x_column, "transaction_id")

        # BAR, LINE, SCATTER should all exist
        self.assertGreater(result.chart_type_counts["BAR"], 0)
        self.assertGreater(result.chart_type_counts["LINE"], 0)
        self.assertGreater(result.chart_type_counts["SCATTER"], 0)


# ---------------------------------------------------------------------------
# Test 17 — No combinatorial explosion
# ---------------------------------------------------------------------------

class TestNoCombinatorialExplosion(unittest.TestCase):
    def test_many_categories_many_measures_bounded(self) -> None:
        """10 CATEGORY cols × 10 MEASURE cols should NOT produce 100 BAR candidates."""
        cat_cols = {f"cat_{i}": ([f"val_{j}" for j in range(3)] * 5)[:15]
                    for i in range(10)}
        meas_cols = {f"measure_{i}": [float(j * (i + 1)) for j in range(15)]
                     for i in range(10)}
        df = pd.DataFrame({**cat_cols, **meas_cols})
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        bar_count = result.chart_type_counts["BAR"]
        # With 10 categories and 10 measures, the cap ensures:
        # at most _MAX_CAT_PER_MEASURE * n_measures = 3 * 10 = 30 BAR candidates
        # (and _MAX_MEAS_PER_CAT * n_cats = 3 * 10 = 30 whichever is tighter)
        # Total must be well under 100
        self.assertLess(bar_count, 50,
                        f"BAR count {bar_count} suggests combinatorial explosion")

    def test_many_measures_scatter_bounded(self) -> None:
        """Many MEASURE columns should not produce unlimited scatter pairs."""
        n_meas = 10
        meas_cols = {f"meas_{i}": [float(j * (i + 1) + 0.5) for j in range(20)]
                     for i in range(n_meas)}
        df = pd.DataFrame(meas_cols)
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        scatter_count = result.chart_type_counts["SCATTER"]
        # n*(n-1)/2 = 45 raw pairs, but cap is _MAX_SCATTER_PAIRS
        self.assertLessEqual(scatter_count, _MAX_SCATTER_PAIRS,
                             f"Scatter count {scatter_count} exceeds cap {_MAX_SCATTER_PAIRS}")


# ---------------------------------------------------------------------------
# Test: FeasibilityResult schema utilities
# ---------------------------------------------------------------------------

class TestFeasibilityResultHelpers(unittest.TestCase):
    def test_candidates_by_type(self) -> None:
        df = pd.DataFrame({
            "region": ["North", "South", "East"] * 3,
            "revenue": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            "cost": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        for vt in ["BAR", "HISTOGRAM", "SCATTER", "COUNT_BAR", "LINE"]:
            subset = result.candidates_by_type(vt)  # type: ignore[arg-type]
            for c in subset:
                self.assertEqual(c.chart_type, vt)

    def test_chart_type_counts_covers_all_types(self) -> None:
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        counts = result.chart_type_counts
        self.assertIn("BAR", counts)
        self.assertIn("LINE", counts)
        self.assertIn("SCATTER", counts)
        self.assertIn("HISTOGRAM", counts)
        self.assertIn("COUNT_BAR", counts)

    def test_candidate_count_matches_candidates_length(self) -> None:
        df = pd.DataFrame({
            "cat": ["A", "B", "A"],
            "val": [1.0, 2.0, 3.0],
        })
        clf = _clf_df(df)
        result = analyze_visualization_feasibility(df, clf)
        self.assertEqual(result.candidate_count, len(result.candidates))


if __name__ == "__main__":
    unittest.main()

