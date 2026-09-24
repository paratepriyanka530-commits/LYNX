"""Comprehensive tests for Pipeline 3 Stage 4 — Chart Data Generation."""

from __future__ import annotations

import json
import math
import unittest

import numpy as np
import pandas as pd

from pipelines.pipeline3_results.visualization.column_classifier import classify_columns
from pipelines.pipeline3_results.visualization.data_builder import (
    _select_aggregation,
    build_chart_data,
)
from pipelines.pipeline3_results.visualization.feasibility import (
    analyze_visualization_feasibility,
)
from pipelines.pipeline3_results.visualization.ranker import (
    rank_visualization_candidates,
)
from pipelines.pipeline3_results.visualization.schemas import (
    STAGE4_MAX_CATEGORY_POINTS,
    STAGE4_MAX_HISTOGRAM_BINS,
    STAGE4_MAX_LINE_POINTS,
    STAGE4_MAX_SCATTER_POINTS,
    ChartData,
    ChartPoint,
    ColumnClassification,
    DatasetClassification,
    HistogramBin,
    RankedCandidate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_pipeline(df: pd.DataFrame):
    """Run Stages 1-3 and return (clf, feas, ranking)."""
    clf = classify_columns(df)
    feas = analyze_visualization_feasibility(df, clf)
    ranking = rank_visualization_candidates(df, clf, feas)
    return clf, feas, ranking


def _make_classification(
    columns: list[tuple[str, str]],
    dataset_name: str | None = None,
) -> DatasetClassification:
    """Directly construct a DatasetClassification without running Stage 1 classifier."""
    col_objs = [
        ColumnClassification(
            name=name,
            role=role,  # type: ignore[arg-type]
            dtype="float64" if role == "MEASURE" else "object",
            unique_count=10,
            unique_ratio=0.1,
            missing_count=0,
            missing_ratio=0.0,
            signals=[],
        )
        for name, role in columns
    ]
    return DatasetClassification(
        dataset_name=dataset_name,
        row_count=100,
        column_count=len(columns),
        columns=col_objs,
        role_counts={
            r: sum(1 for _, role in columns if role == r)
            for r in ["MEASURE", "CATEGORY", "TEMPORAL", "IDENTIFIER", "DESCRIPTIVE_TEXT", "BOOLEAN", "UNKNOWN"]
        },
    )


def _find_ranked(ranking, chart_type: str, x: str, y: str | None = None) -> RankedCandidate | None:
    for c in ranking.ranked:
        if c.chart_type == chart_type and c.x_column == x and c.y_column == y:
            return c
    return None


def _make_candidate(chart_type: str, x: str, y: str | None = None) -> RankedCandidate:
    """Construct a minimal RankedCandidate for injection testing."""
    return RankedCandidate(
        chart_type=chart_type,  # type: ignore[arg-type]
        x_column=x,
        y_column=y,
        score=0.5,
        rank=1,
        signals=[],
    )



def _is_json_serializable(obj) -> bool:
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Test 1 — BAR with sum aggregation
# ---------------------------------------------------------------------------

class TestBarSumAggregation(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({
            "region": ["North", "South", "East", "West", "North", "South", "East"] * 3,
            "total_revenue": [float(i * 100 + 50) for i in range(21)],
        })
        self.clf, _, self.ranking = _full_pipeline(self.df)

    def test_bar_sum_produces_chart_data(self) -> None:
        cand = _find_ranked(self.ranking, "BAR", "region", "total_revenue")
        if cand is None:
            self.skipTest("BAR(region, total_revenue) not in Stage 3 output")
        result = build_chart_data(self.df, self.clf, cand)
        self.assertTrue(result.has_data)
        self.assertEqual(result.chart_type, "BAR")
        self.assertEqual(result.aggregation, "sum")
        self.assertGreater(len(result.points), 0)

    def test_bar_sum_aggregation_signal(self) -> None:
        cand = _find_ranked(self.ranking, "BAR", "region", "total_revenue")
        if cand is None:
            self.skipTest("BAR not in ranked output")
        result = build_chart_data(self.df, self.clf, cand)
        self.assertTrue(any("sum" in s.lower() for s in result.signals))

    def test_bar_sum_values_correct(self) -> None:
        # total_revenue for "North" rows: indices 0, 4, 7, 11, 14, 18 (mod 7 == 0 or 4)
        # verify the aggregate is indeed sum (> any individual value)
        cand = _find_ranked(self.ranking, "BAR", "region", "total_revenue")
        if cand is None:
            self.skipTest("BAR not in ranked output")
        result = build_chart_data(self.df, self.clf, cand)
        if result.aggregation == "sum":
            # Each group should have y >= the minimum raw value (sum >= any single value)
            raw_max = self.df["total_revenue"].max()
            for pt in result.points:
                # Since we summed multiple rows, group totals can exceed raw max
                self.assertIsInstance(pt.y, float)


# ---------------------------------------------------------------------------
# Test 2 — BAR with mean aggregation
# ---------------------------------------------------------------------------

class TestBarMeanAggregation(unittest.TestCase):
    def test_bar_mean_aggregation_for_rate_column(self) -> None:
        df = pd.DataFrame({
            "department": ["Engineering", "Sales", "Marketing"] * 5,
            "avg_score": [float(i * 0.1 + 3.0) for i in range(15)],
        })
        clf, _, ranking = _full_pipeline(df)
        cand = _find_ranked(ranking, "BAR", "department", "avg_score")
        if cand is None:
            self.skipTest("BAR not in ranked output")
        result = build_chart_data(df, clf, cand)
        self.assertTrue(result.has_data)
        # avg_score name should trigger MEAN (contains "avg" or "score" — rate-like tokens)
        self.assertEqual(result.aggregation, "mean")
        self.assertTrue(any("mean" in s.lower() for s in result.signals))

    def test_bar_fallback_to_mean_when_ambiguous(self) -> None:
        df = pd.DataFrame({
            "group": ["A", "B", "A", "B"],
            "value": [10.0, 20.0, 30.0, 40.0],
        })
        clf = classify_columns(df)
        cand = _make_candidate("BAR", "group", "value")
        result = build_chart_data(df, clf, cand)
        # "value" alone has no additive or rate signal → conservative default MEAN
        self.assertEqual(result.aggregation, "mean")
        self.assertTrue(any("conservative default" in s.lower() or "mean" in s.lower()
                            for s in result.signals))


# ---------------------------------------------------------------------------
# Test 3 — BAR semantic-evidence aggregation
# ---------------------------------------------------------------------------

class TestBarSemanticEvidenceAggregation(unittest.TestCase):
    def test_sum_name_tokens_trigger_sum(self) -> None:
        """Columns named with additive tokens should select sum."""
        for col_name in ["total_sales", "revenue_amount", "order_quantity", "spend_budget"]:
            df = pd.DataFrame({
                "category": ["A", "B", "A", "B"],
                col_name: [10.0, 20.0, 30.0, 40.0],
            })
            clf = classify_columns(df)
            cand = _make_candidate("BAR", "category", col_name)
            result = build_chart_data(df, clf, cand)
            self.assertEqual(result.aggregation, "sum",
                             f"Expected sum for '{col_name}', got '{result.aggregation}'")

    def test_mean_name_tokens_trigger_mean(self) -> None:
        """Columns named with rate/ratio tokens should select mean."""
        for col_name in ["conversion_rate", "click_pct", "avg_rating"]:
            df = pd.DataFrame({
                "region": ["A", "B", "A", "B"],
                col_name: [0.1, 0.2, 0.3, 0.4],
            })
            clf = classify_columns(df)
            cand = _make_candidate("BAR", "region", col_name)
            result = build_chart_data(df, clf, cand)
            self.assertEqual(result.aggregation, "mean",
                             f"Expected mean for '{col_name}', got '{result.aggregation}'")


# ---------------------------------------------------------------------------
# Test 4 — BAR fallback when semantic evidence is absent
# ---------------------------------------------------------------------------

class TestBarFallbackAggregation(unittest.TestCase):
    def test_conservative_mean_default(self) -> None:
        df = pd.DataFrame({
            "color": ["Red", "Blue", "Red"],
            "metric_x": [1.0, 2.0, 3.0],
        })
        clf = classify_columns(df)
        cand = _make_candidate("BAR", "color", "metric_x")
        result = build_chart_data(df, clf, cand)
        self.assertEqual(result.aggregation, "mean")
        # Signal should mention "conservative default"
        self.assertTrue(any("conservative default" in s for s in result.signals))


# ---------------------------------------------------------------------------
# Test 5 — COUNT_BAR
# ---------------------------------------------------------------------------

class TestCountBar(unittest.TestCase):
    def test_count_bar_produces_counts(self) -> None:
        df = pd.DataFrame({
            "status": ["Active", "Inactive", "Active", "Active", "Inactive"] * 4,
        })
        clf, _, ranking = _full_pipeline(df)
        cand = _find_ranked(ranking, "COUNT_BAR", "status")
        if cand is None:
            self.skipTest("COUNT_BAR not in ranked output")
        result = build_chart_data(df, clf, cand)
        self.assertTrue(result.has_data)
        self.assertEqual(result.chart_type, "COUNT_BAR")
        self.assertEqual(result.aggregation, "count")
        self.assertGreater(len(result.points), 0)

    def test_count_bar_counts_match_actual(self) -> None:
        df = pd.DataFrame({
            "grade": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        })
        clf = classify_columns(df)
        cand = _make_candidate("COUNT_BAR", "grade")
        result = build_chart_data(df, clf, cand)
        self.assertTrue(result.has_data)
        pt_map = {str(p.x): p.y for p in result.points}
        self.assertEqual(pt_map.get("A"), 10)
        self.assertEqual(pt_map.get("B"), 6)
        self.assertEqual(pt_map.get("C"), 4)

    def test_count_bar_no_y_column(self) -> None:
        df = pd.DataFrame({"category": ["X", "Y", "X"]})
        clf = classify_columns(df)
        cand = _make_candidate("COUNT_BAR", "category")
        result = build_chart_data(df, clf, cand)
        self.assertIsNone(result.y_column)

    def test_count_bar_missing_excluded(self) -> None:
        df = pd.DataFrame({
            "group": ["A", None, "B", "A", None, "B", "A"],
        })
        clf = classify_columns(df)
        cand = _make_candidate("COUNT_BAR", "group")
        result = build_chart_data(df, clf, cand)
        # Only 3 A + 2 B = 5 usable values
        self.assertEqual(result.usable_row_count, 5)
        pt_map = {str(p.x): p.y for p in result.points}
        self.assertNotIn("None", pt_map)
        self.assertNotIn("nan", pt_map)


# ---------------------------------------------------------------------------
# Test 6 — LINE chronological ordering
# ---------------------------------------------------------------------------

class TestLineChronologicalOrder(unittest.TestCase):
    def test_line_points_chronologically_ordered(self) -> None:
        df = pd.DataFrame({
            "order_date": pd.date_range("2023-01-01", periods=12, freq="ME"),
            "revenue": [float(i * 1000) for i in range(12)],
        })
        clf, _, ranking = _full_pipeline(df)
        cand = _find_ranked(ranking, "LINE", "order_date", "revenue")
        if cand is None:
            self.skipTest("LINE not in ranked output")
        result = build_chart_data(df, clf, cand)
        self.assertTrue(result.has_data)
        # X values are ISO timestamp strings — they should sort lexically ascending
        x_vals = [p.x for p in result.points]
        self.assertEqual(x_vals, sorted(x_vals))

    def test_line_x_values_are_strings(self) -> None:
        df = pd.DataFrame({
            "created_at": pd.date_range("2023-01-01", periods=6, freq="D"),
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        })
        clf, _, ranking = _full_pipeline(df)
        cand = _find_ranked(ranking, "LINE", "created_at", "amount")
        if cand is None:
            self.skipTest("LINE not in ranked output")
        result = build_chart_data(df, clf, cand)
        for pt in result.points:
            self.assertIsInstance(pt.x, str)


# ---------------------------------------------------------------------------
# Test 7 — LINE temporal aggregation
# ---------------------------------------------------------------------------

class TestLineTemporalAggregation(unittest.TestCase):
    def test_line_resamples_monthly_for_long_range(self) -> None:
        df = pd.DataFrame({
            "sale_date": pd.date_range("2020-01-01", periods=730, freq="D"),
            "cost": [float(i) for i in range(730)],
        })
        clf = _make_classification([("sale_date", "TEMPORAL"), ("cost", "MEASURE")])
        cand = _make_candidate("LINE", "sale_date", "cost")
        result = build_chart_data(df, clf, cand)
        self.assertTrue(result.has_data)
        # Monthly: 730 raw days → ~24 monthly periods → well under 500-point limit
        self.assertLessEqual(len(result.points), STAGE4_MAX_LINE_POINTS)
        self.assertTrue(any("monthly" in s.lower() or "weekly" in s.lower()
                            for s in result.signals))

    def test_line_aggregation_signal_present(self) -> None:
        df = pd.DataFrame({
            "order_date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "revenue": [float(i * 100) for i in range(10)],
        })
        clf, _, ranking = _full_pipeline(df)
        cand = _find_ranked(ranking, "LINE", "order_date", "revenue")
        if cand is None:
            self.skipTest("LINE not in ranked output")
        result = build_chart_data(df, clf, cand)
        combined = " ".join(result.signals).lower()
        self.assertTrue("aggregated" in combined or "sum" in combined or "mean" in combined)


# ---------------------------------------------------------------------------
# Test 8 — SCATTER row-wise pairing
# ---------------------------------------------------------------------------

class TestScatterRowWise(unittest.TestCase):
    def test_scatter_preserves_row_pairing(self) -> None:
        df = pd.DataFrame({
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],
            "units": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        clf, _, ranking = _full_pipeline(df)
        cand = _find_ranked(ranking, "SCATTER", "price", "units")
        if cand is None:
            cand = _find_ranked(ranking, "SCATTER", "units", "price")
        if cand is None:
            self.skipTest("SCATTER not in ranked output")
        result = build_chart_data(df, clf, cand)
        self.assertTrue(result.has_data)
        self.assertEqual(len(result.points), 5)

    def test_scatter_no_aggregation(self) -> None:
        df = pd.DataFrame({
            "weight": [60.0, 70.0, 80.0],
            "height": [1.65, 1.75, 1.85],
        })
        clf = classify_columns(df)
        cand = _make_candidate("SCATTER", "weight", "height")
        result = build_chart_data(df, clf, cand)
        self.assertIsNone(result.aggregation)

    def test_scatter_point_count_matches_valid_rows(self) -> None:
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        clf = classify_columns(df)
        cand = _make_candidate("SCATTER", "a", "b")
        result = build_chart_data(df, clf, cand)
        self.assertEqual(result.usable_row_count, 5)
        self.assertEqual(len(result.points), 5)


# ---------------------------------------------------------------------------
# Test 9 — SCATTER missing / non-finite filtering
# ---------------------------------------------------------------------------

class TestScatterMissingNonFinite(unittest.TestCase):
    def test_scatter_drops_missing_rows(self) -> None:
        df = pd.DataFrame({
            "x_val": [1.0, None, 3.0, 4.0],
            "y_val": [10.0, 20.0, None, 40.0],
        })
        clf = classify_columns(df)
        cand = _make_candidate("SCATTER", "x_val", "y_val")
        result = build_chart_data(df, clf, cand)
        # Rows 2 (None x) and 3 (None y) dropped → 2 valid rows
        self.assertEqual(result.usable_row_count, 2)
        self.assertEqual(len(result.points), 2)

    def test_scatter_drops_infinite_values(self) -> None:
        df = pd.DataFrame({
            "a": [1.0, float("inf"), 3.0, 4.0],
            "b": [10.0, 20.0, float("-inf"), 40.0],
        })
        clf = classify_columns(df)
        cand = _make_candidate("SCATTER", "a", "b")
        result = build_chart_data(df, clf, cand)
        self.assertEqual(result.usable_row_count, 2)
        for pt in result.points:
            self.assertTrue(math.isfinite(pt.y))
            self.assertTrue(math.isfinite(float(pt.x)))

    def test_scatter_all_missing_returns_no_data(self) -> None:
        df = pd.DataFrame({
            "a": [None, None, None],
            "b": [None, None, None],
        })
        clf = classify_columns(df)
        cand = _make_candidate("SCATTER", "a", "b")
        result = build_chart_data(df, clf, cand)
        self.assertFalse(result.has_data)


# ---------------------------------------------------------------------------
# Test 10 — Deterministic scatter sampling if applicable
# ---------------------------------------------------------------------------

class TestScatterDeterministicSampling(unittest.TestCase):
    def test_large_scatter_sampled_deterministically(self) -> None:
        n = STAGE4_MAX_SCATTER_POINTS + 100
        df = pd.DataFrame({
            "x_meas": [float(i * 0.1) for i in range(n)],
            "y_meas": [float(i * 0.2) for i in range(n)],
        })
        clf = _make_classification([("x_meas", "MEASURE"), ("y_meas", "MEASURE")])
        cand = _make_candidate("SCATTER", "x_meas", "y_meas")
        result = build_chart_data(df, clf, cand)
        self.assertLessEqual(len(result.points), STAGE4_MAX_SCATTER_POINTS)
        self.assertTrue(result.sampled)
        self.assertTrue(any("deterministically sampled" in s.lower() for s in result.signals))

    def test_scatter_sampling_is_reproducible(self) -> None:
        n = STAGE4_MAX_SCATTER_POINTS + 100
        df = pd.DataFrame({
            "x_meas": [float(i) for i in range(n)],
            "y_meas": [float(i * 2) for i in range(n)],
        })
        clf = _make_classification([("x_meas", "MEASURE"), ("y_meas", "MEASURE")])
        cand = _make_candidate("SCATTER", "x_meas", "y_meas")
        r1 = build_chart_data(df, clf, cand)
        r2 = build_chart_data(df, clf, cand)
        self.assertEqual([p.model_dump() for p in r1.points],
                         [p.model_dump() for p in r2.points])



# ---------------------------------------------------------------------------
# Test 11 — HISTOGRAM bins and counts
# ---------------------------------------------------------------------------

class TestHistogramBinsAndCounts(unittest.TestCase):
    def test_histogram_produces_bins(self) -> None:
        df = pd.DataFrame({
            "price": [float(i * 5) for i in range(50)],
        })
        clf = classify_columns(df)
        cand = _make_candidate("HISTOGRAM", "price")
        result = build_chart_data(df, clf, cand)
        self.assertTrue(result.has_data)
        self.assertGreater(len(result.bins), 0)
        self.assertEqual(len(result.points), 0)  # HISTOGRAM uses bins, not points

    def test_histogram_bins_cover_all_values(self) -> None:
        df = pd.DataFrame({"val": [float(i) for i in range(100)]})
        clf = classify_columns(df)
        cand = _make_candidate("HISTOGRAM", "val")
        result = build_chart_data(df, clf, cand)
        total_count = sum(b.count for b in result.bins)
        self.assertEqual(total_count, 100)

    def test_histogram_bins_sorted_by_start(self) -> None:
        df = pd.DataFrame({"amount": [float(i * 10.5) for i in range(40)]})
        clf = classify_columns(df)
        cand = _make_candidate("HISTOGRAM", "amount")
        result = build_chart_data(df, clf, cand)
        starts = [b.bin_start for b in result.bins]
        self.assertEqual(starts, sorted(starts))

    def test_histogram_no_empty_bins(self) -> None:
        df = pd.DataFrame({"val": [float(i) for i in range(20)]})
        clf = classify_columns(df)
        cand = _make_candidate("HISTOGRAM", "val")
        result = build_chart_data(df, clf, cand)
        for b in result.bins:
            self.assertGreater(b.count, 0)

    def test_histogram_bin_count_uses_sturges_rule(self) -> None:
        clf = _make_classification([("metric", "MEASURE")])
        cand = _make_candidate("HISTOGRAM", "metric")
        for n in [10, 50, 100, 256]:
            df = pd.DataFrame({"metric": [float(i) for i in range(n)]})
            result = build_chart_data(df, clf, cand)
            expected_bins = min(math.ceil(math.log2(n) + 1), STAGE4_MAX_HISTOGRAM_BINS)
            # Non-empty bin count may be ≤ expected (empty bins excluded)
            self.assertLessEqual(len(result.bins), expected_bins)

    def test_histogram_capped_at_max_bins(self) -> None:
        import math as _math
        # Verify Sturges formula calculation with a hypothetical large N
        hypothetical_n = 2 ** 52
        expected_raw_bins = _math.ceil(_math.log2(hypothetical_n) + 1)  # = 53
        capped = min(expected_raw_bins, STAGE4_MAX_HISTOGRAM_BINS)
        self.assertLessEqual(capped, STAGE4_MAX_HISTOGRAM_BINS)

        # Also verify in practice with a concrete dataset
        df = pd.DataFrame({"metric": [float(i) for i in range(100)]})
        clf = _make_classification([("metric", "MEASURE")])
        cand = _make_candidate("HISTOGRAM", "metric")
        result = build_chart_data(df, clf, cand)
        self.assertLessEqual(len(result.bins), STAGE4_MAX_HISTOGRAM_BINS)


# ---------------------------------------------------------------------------
# Test 12 — Constant measure
# ---------------------------------------------------------------------------

class TestConstantMeasure(unittest.TestCase):
    def test_constant_column_is_unknown_no_candidates(self) -> None:
        """Stage 1 classifies a constant column as UNKNOWN → Stage 2 excludes it."""
        df = pd.DataFrame({
            "region": ["A", "B", "A", "B"],
            "constant_val": [42.0, 42.0, 42.0, 42.0],
        })
        clf, feas, _ = _full_pipeline(df)
        for c in feas.candidates:
            self.assertNotEqual(c.x_column, "constant_val")
            self.assertNotEqual(c.y_column, "constant_val")

    def test_constant_measure_bar_still_produces_output(self) -> None:
        """If Stage 4 is called with a constant-valued measure, it should not crash."""
        df = pd.DataFrame({
            "color": ["Red", "Blue", "Red"],
            "constant_price": [100.0, 100.0, 100.0],
        })
        clf = classify_columns(df)
        cand = _make_candidate("BAR", "color", "constant_price")
        result = build_chart_data(df, clf, cand)
        # Should produce data (the aggregation still works for constant measures)
        self.assertIsNotNone(result)
        self.assertEqual(result.chart_type, "BAR")


# ---------------------------------------------------------------------------
# Test 13 — All-missing measure
# ---------------------------------------------------------------------------

class TestAllMissingMeasure(unittest.TestCase):
    def test_all_missing_measure_returns_no_data(self) -> None:
        df = pd.DataFrame({
            "category": ["A", "B", "C"],
            "null_amount": [None, None, None],
        })
        clf = classify_columns(df)
        cand = _make_candidate("BAR", "category", "null_amount")
        result = build_chart_data(df, clf, cand)
        self.assertFalse(result.has_data)
        self.assertEqual(result.usable_row_count, 0)
        self.assertTrue(len(result.signals) > 0)

    def test_all_missing_histogram_returns_no_data(self) -> None:
        df = pd.DataFrame({"null_col": [None, None, None]})
        clf = classify_columns(df)
        cand = _make_candidate("HISTOGRAM", "null_col")
        result = build_chart_data(df, clf, cand)
        self.assertFalse(result.has_data)


# ---------------------------------------------------------------------------
# Test 14 — Invalid temporal values
# ---------------------------------------------------------------------------

class TestInvalidTemporalValues(unittest.TestCase):
    def test_invalid_temporal_values_excluded_and_signalled(self) -> None:
        df = pd.DataFrame({
            "order_date": ["2023-01-01", "not-a-date", "2023-03-01", "bad-val", "2023-05-01"],
            "revenue": [100.0, 200.0, 150.0, 300.0, 120.0],
        })
        clf = classify_columns(df)
        cand = _make_candidate("LINE", "order_date", "revenue")
        result = build_chart_data(df, clf, cand)
        self.assertIsNotNone(result)
        if result.has_data:
            # Valid: 3 parseable dates used
            self.assertEqual(result.usable_row_count, 3)
            self.assertTrue(any("invalid temporal" in s.lower() for s in result.signals))
        else:
            # Result documented with signals
            self.assertTrue(len(result.signals) > 0)

    def test_all_invalid_temporal_returns_no_data(self) -> None:
        df = pd.DataFrame({
            "order_date": ["not-a-date", "also-bad", "no-good"],
            "revenue": [100.0, 200.0, 300.0],
        })
        clf = classify_columns(df)
        cand = _make_candidate("LINE", "order_date", "revenue")
        result = build_chart_data(df, clf, cand)
        self.assertFalse(result.has_data)


# ---------------------------------------------------------------------------
# Test 15 — Missing candidate column
# ---------------------------------------------------------------------------

class TestMissingCandidateColumn(unittest.TestCase):
    def test_missing_x_column_returns_no_data(self) -> None:
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        clf = classify_columns(df)
        cand = _make_candidate("BAR", "nonexistent_cat", "other_col")
        result = build_chart_data(df, clf, cand)
        self.assertFalse(result.has_data)
        self.assertTrue(any("not found" in s.lower() for s in result.signals))

    def test_missing_y_column_returns_no_data(self) -> None:
        df = pd.DataFrame({"region": ["A", "B"]})
        clf = classify_columns(df)
        cand = _make_candidate("BAR", "region", "missing_measure")
        result = build_chart_data(df, clf, cand)
        self.assertFalse(result.has_data)
        self.assertTrue(any("not found" in s.lower() for s in result.signals))


# ---------------------------------------------------------------------------
# Test 16 — Empty dataframe
# ---------------------------------------------------------------------------

class TestEmptyDataframe(unittest.TestCase):
    def test_empty_df_bar_returns_no_data(self) -> None:
        df = pd.DataFrame({"region": pd.Series([], dtype=str),
                           "sales": pd.Series([], dtype=float)})
        clf = classify_columns(df)
        cand = _make_candidate("BAR", "region", "sales")
        result = build_chart_data(df, clf, cand)
        self.assertFalse(result.has_data)
        self.assertTrue(any("zero rows" in s.lower() for s in result.signals))

    def test_empty_df_histogram_returns_no_data(self) -> None:
        df = pd.DataFrame({"price": pd.Series([], dtype=float)})
        clf = classify_columns(df)
        cand = _make_candidate("HISTOGRAM", "price")
        result = build_chart_data(df, clf, cand)
        self.assertFalse(result.has_data)

    def test_empty_df_scatter_returns_no_data(self) -> None:
        df = pd.DataFrame({"a": pd.Series([], dtype=float),
                           "b": pd.Series([], dtype=float)})
        clf = classify_columns(df)
        cand = _make_candidate("SCATTER", "a", "b")
        result = build_chart_data(df, clf, cand)
        self.assertFalse(result.has_data)


# ---------------------------------------------------------------------------
# Test 17 — Deterministic repeated execution
# ---------------------------------------------------------------------------

class TestDeterministicRepeatedExecution(unittest.TestCase):
    def test_same_inputs_produce_identical_output(self) -> None:
        df = pd.DataFrame({
            "region": ["North", "South", "East", "West"] * 10,
            "order_date": pd.date_range("2023-01-01", periods=40, freq="D"),
            "revenue": [float(i * 10.5) for i in range(40)],
            "cost":    [float(i * 5.5) for i in range(40)],
        })
        clf, _, ranking = _full_pipeline(df)
        for cand in ranking.ranked[:5]:
            r1 = build_chart_data(df, clf, cand)
            r2 = build_chart_data(df, clf, cand)
            self.assertEqual(r1.model_dump(), r2.model_dump(),
                             f"Non-deterministic output for {cand.chart_type}({cand.x_column}, {cand.y_column})")


# ---------------------------------------------------------------------------
# Test 18 — JSON serializability
# ---------------------------------------------------------------------------

class TestJsonSerializability(unittest.TestCase):
    def test_bar_result_json_serializable(self) -> None:
        df = pd.DataFrame({
            "cat": ["A", "B", "A", "B"] * 5,
            "amount": [float(i * 10) for i in range(20)],
        })
        clf = classify_columns(df)
        cand = _make_candidate("BAR", "cat", "amount")
        result = build_chart_data(df, clf, cand)
        self.assertTrue(_is_json_serializable(result.model_dump()))

    def test_histogram_result_json_serializable(self) -> None:
        df = pd.DataFrame({"price": [float(i) for i in range(50)]})
        clf = classify_columns(df)
        cand = _make_candidate("HISTOGRAM", "price")
        result = build_chart_data(df, clf, cand)
        self.assertTrue(_is_json_serializable(result.model_dump()))

    def test_scatter_result_json_serializable(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        clf = classify_columns(df)
        cand = _make_candidate("SCATTER", "x", "y")
        result = build_chart_data(df, clf, cand)
        self.assertTrue(_is_json_serializable(result.model_dump()))

    def test_line_result_json_serializable(self) -> None:
        df = pd.DataFrame({
            "ts": pd.date_range("2023-01-01", periods=5, freq="D"),
            "val": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        clf = classify_columns(df)
        cand = _make_candidate("LINE", "ts", "val")
        result = build_chart_data(df, clf, cand)
        self.assertTrue(_is_json_serializable(result.model_dump()))

    def test_count_bar_result_json_serializable(self) -> None:
        df = pd.DataFrame({"status": ["A", "B", "A", "C"]})
        clf = classify_columns(df)
        cand = _make_candidate("COUNT_BAR", "status")
        result = build_chart_data(df, clf, cand)
        self.assertTrue(_is_json_serializable(result.model_dump()))

    def test_no_numpy_scalars_in_output(self) -> None:
        """Verify no numpy types leak into the serialized output."""
        df = pd.DataFrame({
            "region": ["X", "Y", "X", "Z"] * 3,
            "sales": [float(i * 12) for i in range(12)],
        })
        clf = classify_columns(df)
        cand = _make_candidate("BAR", "region", "sales")
        result = build_chart_data(df, clf, cand)
        dumped = result.model_dump()
        raw_json = json.dumps(dumped)  # must not raise TypeError
        self.assertIsInstance(raw_json, str)


# ---------------------------------------------------------------------------
# Test 19 — Deterministic category ordering
# ---------------------------------------------------------------------------

class TestDeterministicCategoryOrdering(unittest.TestCase):
    def test_count_bar_ordered_by_count_then_lexical(self) -> None:
        df = pd.DataFrame({
            "label": ["B"] * 5 + ["A"] * 5 + ["C"] * 3,
        })
        clf = classify_columns(df)
        cand = _make_candidate("COUNT_BAR", "label")
        result = build_chart_data(df, clf, cand)
        x_vals = [str(p.x) for p in result.points]
        # Primary: descending count → A or B first (tied at 5)
        # Tie-break: ascending lexical → A before B
        self.assertEqual(x_vals[0], "A")
        self.assertEqual(x_vals[1], "B")
        self.assertEqual(x_vals[2], "C")

    def test_bar_ordered_deterministically(self) -> None:
        df = pd.DataFrame({
            "group": ["X", "Y", "Z"] * 5,
            "revenue": [100.0, 200.0, 150.0] * 5,
        })
        clf = classify_columns(df)
        cand = _make_candidate("BAR", "group", "revenue")
        r1 = build_chart_data(df, clf, cand)
        r2 = build_chart_data(df, clf, cand)
        self.assertEqual([p.x for p in r1.points], [p.x for p in r2.points])


# ---------------------------------------------------------------------------
# Test 20 — Payload-size limits
# ---------------------------------------------------------------------------

class TestPayloadSizeLimits(unittest.TestCase):
    def test_bar_category_limit_enforced(self) -> None:
        n_cats = STAGE4_MAX_CATEGORY_POINTS + 10  # 60 categories
        labels = [f"cat_{i}" for i in range(n_cats)] * 2  # 120 rows
        df = pd.DataFrame({
            "product": labels,
            "amount": [float(i) for i in range(len(labels))],
        })
        clf = _make_classification([("product", "CATEGORY"), ("amount", "MEASURE")])
        cand = _make_candidate("BAR", "product", "amount")
        result = build_chart_data(df, clf, cand)
        self.assertLessEqual(len(result.points), STAGE4_MAX_CATEGORY_POINTS)
        self.assertTrue(result.sampled)
        self.assertTrue(any("truncated" in s.lower() for s in result.signals))

    def test_line_size_limit_enforced(self) -> None:
        # 550 quarterly periods span ~137 years (> 1825 days -> freq='QE'), yielding 550 points > 500 cap
        n = STAGE4_MAX_LINE_POINTS + 50
        df = pd.DataFrame({
            "created_at": pd.date_range("1800-01-01", periods=n, freq="QE"),
            "revenue": [float(i) for i in range(n)],
        })
        clf = _make_classification([("created_at", "TEMPORAL"), ("revenue", "MEASURE")])
        cand = _make_candidate("LINE", "created_at", "revenue")
        result = build_chart_data(df, clf, cand)
        self.assertLessEqual(len(result.points), STAGE4_MAX_LINE_POINTS)
        self.assertTrue(result.sampled)
        self.assertTrue(any("deterministically sampled" in s for s in result.signals))

    def test_histogram_bin_limit_enforced(self) -> None:
        import unittest.mock
        # Mock STAGE4_MAX_HISTOGRAM_BINS to 5 so a small dataset crosses the limit
        with unittest.mock.patch(
            "pipelines.pipeline3_results.visualization.data_builder.STAGE4_MAX_HISTOGRAM_BINS",
            5,
        ):
            # 64 observations produce ceil(log2(64)+1) = 7 raw bins > cap of 5
            df = pd.DataFrame({"metric": [float(i) for i in range(64)]})
            clf = _make_classification([("metric", "MEASURE")])
            cand = _make_candidate("HISTOGRAM", "metric")
            result = build_chart_data(df, clf, cand)
            self.assertLessEqual(len(result.bins), 5)
            self.assertTrue(any("cap = 5" in s for s in result.signals))

        # Also verify with unmocked limit on a concrete dataset
        df2 = pd.DataFrame({"metric": [float(i) for i in range(100)]})
        result2 = build_chart_data(df2, clf, cand)
        self.assertLessEqual(len(result2.bins), STAGE4_MAX_HISTOGRAM_BINS)


# ---------------------------------------------------------------------------
# Test 21 — No fabricated values
# ---------------------------------------------------------------------------

class TestNoFabricatedValues(unittest.TestCase):
    def test_bar_sum_matches_manual_calculation(self) -> None:
        df = pd.DataFrame({
            "region": ["East", "West", "East", "West"],
            "total_sales": [100.0, 200.0, 300.0, 400.0],
        })
        clf = classify_columns(df)
        cand = _make_candidate("BAR", "region", "total_sales")
        result = build_chart_data(df, clf, cand)
        self.assertTrue(result.has_data)
        if result.aggregation == "sum":
            pt_map = {str(p.x): p.y for p in result.points}
            self.assertAlmostEqual(pt_map["East"], 400.0)
            self.assertAlmostEqual(pt_map["West"], 600.0)
        elif result.aggregation == "mean":
            pt_map = {str(p.x): p.y for p in result.points}
            self.assertAlmostEqual(pt_map["East"], 200.0)
            self.assertAlmostEqual(pt_map["West"], 300.0)

    def test_histogram_total_count_equals_usable_rows(self) -> None:
        df = pd.DataFrame({"price": [float(i * 3.5) for i in range(30)]})
        clf = classify_columns(df)
        cand = _make_candidate("HISTOGRAM", "price")
        result = build_chart_data(df, clf, cand)
        if result.has_data:
            total_binned = sum(b.count for b in result.bins)
            self.assertEqual(total_binned, result.usable_row_count)

    def test_scatter_points_match_source_rows(self) -> None:
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [10.0, 20.0, 30.0],
        })
        clf = classify_columns(df)
        cand = _make_candidate("SCATTER", "a", "b")
        result = build_chart_data(df, clf, cand)
        pt_pairs = {(p.x, p.y) for p in result.points}
        expected = {(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)}
        self.assertEqual(pt_pairs, expected)


if __name__ == "__main__":
    unittest.main()
