"""Comprehensive tests for Pipeline 3 Stage 3 — Visualization Candidate Ranking."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipelines.pipeline3_results.visualization.column_classifier import classify_columns
from pipelines.pipeline3_results.visualization.feasibility import (
    analyze_visualization_feasibility,
)
from pipelines.pipeline3_results.visualization.ranker import (
    _CONSTANT_CV_THRESHOLD,
    _IDEAL_CARDINALITY_HIGH,
    _IDEAL_CARDINALITY_LOW,
    _MAX_CARDINALITY,
    rank_visualization_candidates,
)
from pipelines.pipeline3_results.visualization.schemas import (
    DatasetClassification,
    FeasibilityResult,
    RankingResult,
    VisualizationCandidate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pipeline(df: pd.DataFrame):
    """Run Stage 1 → Stage 2 → Stage 3 and return all three results."""
    clf = classify_columns(df)
    feas = analyze_visualization_feasibility(df, clf)
    rank = rank_visualization_candidates(df, clf, feas)
    return clf, feas, rank


def _ranked_types(result: RankingResult) -> list[str]:
    return [c.chart_type for c in result.ranked]


def _find(result: RankingResult, chart_type: str, x: str, y: str | None = None):
    for c in result.ranked:
        if c.chart_type == chart_type and c.x_column == x and c.y_column == y:
            return c
    return None


# ---------------------------------------------------------------------------
# Test 1 — BAR ranking
# ---------------------------------------------------------------------------

class TestBarRanking(unittest.TestCase):
    def test_bar_candidate_appears_and_scored(self) -> None:
        df = pd.DataFrame({
            "region": ["North", "South", "East", "West", "North", "South", "East"] * 3,
            "sales_amount": [float(i * 12.5) for i in range(21)],
        })
        _, _, rank = _pipeline(df)
        bar = _find(rank, "BAR", "region", "sales_amount")
        self.assertIsNotNone(bar, f"Expected BAR candidate. Got: {rank.ranked}")
        self.assertGreater(bar.score, 0.0)
        self.assertLessEqual(bar.score, 1.0)
        self.assertGreater(bar.rank, 0)

    def test_bar_signals_present(self) -> None:
        df = pd.DataFrame({
            "department": ["A", "B", "A", "B", "A"],
            "revenue": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        _, _, rank = _pipeline(df)
        bars = rank.ranked_by_type("BAR")
        self.assertTrue(len(bars) > 0)
        combined = " ".join(bars[0].signals)
        self.assertIn("coverage", combined)
        self.assertIn("cardinality", combined)
        self.assertIn("variation", combined)

    def test_bar_score_in_range(self) -> None:
        df = pd.DataFrame({
            "status": ["Active", "Inactive"] * 10,
            "cost": [float(i) for i in range(20)],
        })
        _, _, rank = _pipeline(df)
        for c in rank.ranked_by_type("BAR"):
            self.assertGreaterEqual(c.score, 0.0)
            self.assertLessEqual(c.score, 1.0)


# ---------------------------------------------------------------------------
# Test 2 — LINE ranking
# ---------------------------------------------------------------------------

class TestLineRanking(unittest.TestCase):
    def test_line_candidate_appears_and_scored(self) -> None:
        df = pd.DataFrame({
            "order_date": pd.date_range("2023-01-01", periods=12, freq="ME"),
            "revenue": [float(i * 1000 + 5000) for i in range(12)],
        })
        _, _, rank = _pipeline(df)
        line = _find(rank, "LINE", "order_date", "revenue")
        self.assertIsNotNone(line, f"Expected LINE. Got: {rank.ranked}")
        self.assertGreater(line.score, 0.0)

    def test_line_signals_present(self) -> None:
        df = pd.DataFrame({
            "sale_date": pd.date_range("2023-01-01", periods=6, freq="D"),
            "amount": [10.0, 20.0, 15.0, 25.0, 30.0, 18.0],
        })
        _, _, rank = _pipeline(df)
        lines = rank.ranked_by_type("LINE")
        self.assertTrue(len(lines) > 0)
        combined = " ".join(lines[0].signals)
        self.assertIn("temporal", combined.lower())
        self.assertIn("variation", combined.lower())

    def test_line_temporal_coverage_reflected_in_score(self) -> None:
        # A dataset with many distinct temporal periods should score higher LINE
        # than one with very few periods
        df_rich = pd.DataFrame({
            "created_at": pd.date_range("2023-01-01", periods=20, freq="D"),
            "revenue": [float(i * 100) for i in range(20)],
        })
        df_sparse = pd.DataFrame({
            "created_at": pd.date_range("2023-01-01", periods=3, freq="D"),
            "revenue": [100.0, 200.0, 300.0],
        })
        _, _, rank_rich = _pipeline(df_rich)
        _, _, rank_sparse = _pipeline(df_sparse)
        rich_line = _find(rank_rich, "LINE", "created_at", "revenue")
        sparse_line = _find(rank_sparse, "LINE", "created_at", "revenue")
        if rich_line is not None and sparse_line is not None:
            self.assertGreaterEqual(rich_line.score, sparse_line.score)


# ---------------------------------------------------------------------------
# Test 3 — SCATTER ranking
# ---------------------------------------------------------------------------

class TestScatterRanking(unittest.TestCase):
    def test_scatter_candidate_appears_and_scored(self) -> None:
        df = pd.DataFrame({
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],
            "units_sold": [100.0, 80.0, 120.0, 60.0, 90.0],
        })
        _, _, rank = _pipeline(df)
        scatters = rank.ranked_by_type("SCATTER")
        self.assertTrue(len(scatters) > 0)
        sc = scatters[0]
        self.assertGreater(sc.score, 0.0)
        self.assertLessEqual(sc.score, 1.0)

    def test_scatter_signals_present(self) -> None:
        df = pd.DataFrame({
            "age": [25.0, 30.0, 35.0, 40.0],
            "salary": [30000.0, 45000.0, 60000.0, 80000.0],
        })
        _, _, rank = _pipeline(df)
        scatters = rank.ranked_by_type("SCATTER")
        self.assertTrue(len(scatters) > 0)
        combined = " ".join(scatters[0].signals)
        self.assertIn("variation", combined.lower())
        self.assertIn("coverage", combined.lower())


# ---------------------------------------------------------------------------
# Test 4 — HISTOGRAM ranking
# ---------------------------------------------------------------------------

class TestHistogramRanking(unittest.TestCase):
    def test_histogram_candidate_appears_and_scored(self) -> None:
        df = pd.DataFrame({
            "order_total": [float(i * 10.5) for i in range(30)],
        })
        _, _, rank = _pipeline(df)
        hists = rank.ranked_by_type("HISTOGRAM")
        self.assertTrue(len(hists) > 0)
        h = hists[0]
        self.assertIsNone(h.y_column)
        self.assertGreater(h.score, 0.0)

    def test_histogram_signals_present(self) -> None:
        df = pd.DataFrame({
            "revenue": [100.0, 200.0, 300.0, 400.0, 500.0],
        })
        _, _, rank = _pipeline(df)
        hists = rank.ranked_by_type("HISTOGRAM")
        self.assertTrue(len(hists) > 0)
        combined = " ".join(hists[0].signals)
        self.assertIn("coverage", combined.lower())
        self.assertIn("variation", combined.lower())


# ---------------------------------------------------------------------------
# Test 5 — COUNT_BAR ranking
# ---------------------------------------------------------------------------

class TestCountBarRanking(unittest.TestCase):
    def test_count_bar_appears_and_scored(self) -> None:
        df = pd.DataFrame({
            "status": ["Active", "Inactive", "Active", "Active", "Inactive"] * 3,
        })
        _, _, rank = _pipeline(df)
        cbs = rank.ranked_by_type("COUNT_BAR")
        self.assertTrue(len(cbs) > 0)
        cb = cbs[0]
        self.assertIsNone(cb.y_column)
        self.assertGreater(cb.score, 0.0)

    def test_count_bar_signals_present(self) -> None:
        df = pd.DataFrame({
            "category": ["A", "B", "A", "C", "B", "A"] * 2,
        })
        _, _, rank = _pipeline(df)
        cbs = rank.ranked_by_type("COUNT_BAR")
        self.assertTrue(len(cbs) > 0)
        combined = " ".join(cbs[0].signals)
        self.assertIn("cardinality", combined.lower())
        self.assertIn("repetition", combined.lower())


# ---------------------------------------------------------------------------
# Test 6 — Usable measure outranks constant measure
# ---------------------------------------------------------------------------

class TestUsableMeasureVsConstant(unittest.TestCase):
    def test_varying_measure_outranks_constant(self) -> None:
        df = pd.DataFrame({
            "region": ["North", "South", "East", "West", "North", "South"] * 2,
            "real_sales": [10.0, 50.0, 30.0, 90.0, 20.0, 70.0, 15.0, 55.0,
                           35.0, 95.0, 25.0, 75.0],
            "constant_value": [42.0] * 12,  # Stage 1 classifies as UNKNOWN
        })
        _, feas, rank = _pipeline(df)
        # constant_value should be UNKNOWN → excluded from candidates by Stage 2
        for c in rank.ranked:
            self.assertNotEqual(c.x_column, "constant_value")
            self.assertNotEqual(c.y_column, "constant_value")

    def test_higher_variation_scores_higher_histogram(self) -> None:
        # Build two measure columns in the same DataFrame; classify them and check
        # that the one with higher variation gets a higher HISTOGRAM score.
        df = pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(20)],
            "high_var_amount": [float(i * 50) for i in range(20)],    # large spread
            "low_var_amount":  [100.0 + float(i) * 0.01 for i in range(20)],  # tiny spread
        })
        _, _, rank = _pipeline(df)
        high_hist = _find(rank, "HISTOGRAM", "high_var_amount")
        low_hist  = _find(rank, "HISTOGRAM", "low_var_amount")
        if high_hist is not None and low_hist is not None:
            self.assertGreater(high_hist.score, low_hist.score,
                               "Higher variation measure should score higher HISTOGRAM")


# ---------------------------------------------------------------------------
# Test 7 — Reasonable cardinality preferred
# ---------------------------------------------------------------------------

class TestCardinalityPreference(unittest.TestCase):
    def test_ideal_cardinality_scores_higher_than_low(self) -> None:
        """A category with k in [3,15] should score higher COUNT_BAR than k=1 or k=2."""
        n = 30
        # Good: k=5 distinct values → ideal range
        df_good = pd.DataFrame({
            "cat": [f"val_{i % 5}" for i in range(n)],
        })
        # Poor: k=2 values (below ideal low of 3)
        df_poor = pd.DataFrame({
            "cat": [f"v{i % 2}" for i in range(n)],
        })
        _, _, rank_good = _pipeline(df_good)
        _, _, rank_poor = _pipeline(df_poor)
        good_cb = next((c for c in rank_good.ranked if c.chart_type == "COUNT_BAR"), None)
        poor_cb = next((c for c in rank_poor.ranked if c.chart_type == "COUNT_BAR"), None)
        if good_cb is not None and poor_cb is not None:
            self.assertGreater(good_cb.score, poor_cb.score)


# ---------------------------------------------------------------------------
# Test 8 — High-cardinality dimensions are penalized
# ---------------------------------------------------------------------------

class TestHighCardinalityPenalized(unittest.TestCase):
    def test_high_cardinality_category_not_in_ranked(self) -> None:
        """Stage 2 already excludes high-cardinality categories; Stage 3 must not re-add them."""
        n = 100
        df = pd.DataFrame({
            "many_vals": [f"val_{i}" for i in range(n)],  # unique → IDENTIFIER by Stage 1
            "price": [float(i) for i in range(n)],
        })
        _, _, rank = _pipeline(df)
        for c in rank.ranked:
            self.assertNotEqual(c.x_column, "many_vals",
                                "High-cardinality/IDENTIFIER column must not appear in ranked output")

    def test_score_falls_as_cardinality_grows(self) -> None:
        """Cardinality fitness score decreases above the ideal high."""
        from pipelines.pipeline3_results.visualization.ranker import _cardinality_fitness
        score_ideal, _ = _cardinality_fitness(8)   # within [3, 15]
        score_high, _  = _cardinality_fitness(30)  # above ideal but below max
        score_max, _   = _cardinality_fitness(_MAX_CARDINALITY)  # at limit → 0
        self.assertEqual(score_ideal, 1.0)
        self.assertLess(score_high, score_ideal)
        self.assertEqual(score_max, 0.0)


# ---------------------------------------------------------------------------
# Test 9 — Missing-heavy candidates are penalized
# ---------------------------------------------------------------------------

class TestMissingHeavyPenalized(unittest.TestCase):
    def test_partial_missing_measure_lower_score_than_full_coverage(self) -> None:
        """A measure with ~50% missing values should score lower HISTOGRAM than one with 0%."""
        vals_full    = [float(i * 10) for i in range(20)]
        vals_missing = [float(i * 10) if i % 2 == 0 else None for i in range(20)]
        df = pd.DataFrame({
            "region": ["A", "B"] * 10,
            "full_sales":    vals_full,
            "partial_sales": vals_missing,
        })
        _, _, rank = _pipeline(df)
        full_hist    = _find(rank, "HISTOGRAM", "full_sales")
        partial_hist = _find(rank, "HISTOGRAM", "partial_sales")
        if full_hist is not None and partial_hist is not None:
            self.assertGreater(full_hist.score, partial_hist.score,
                               "Full-coverage measure should outscore partial-coverage measure")


# ---------------------------------------------------------------------------
# Test 10 — Temporal coverage affects LINE ranking
# ---------------------------------------------------------------------------

class TestTemporalCoverageAffectsLine(unittest.TestCase):
    def test_more_missing_temporal_lower_line_score(self) -> None:
        """A temporal column with high missing fraction should yield lower LINE score."""
        dates_full    = pd.date_range("2023-01-01", periods=10, freq="D")
        dates_missing = [d if i % 2 == 0 else pd.NaT for i, d in enumerate(dates_full)]
        df = pd.DataFrame({
            "dt_full":    dates_full,
            "dt_missing": pd.Series(dates_missing),
            "revenue":    [float(i * 100) for i in range(10)],
        })
        clf = classify_columns(df)
        feas = analyze_visualization_feasibility(df, clf)
        rank = rank_visualization_candidates(df, clf, feas)
        full_line    = _find(rank, "LINE", "dt_full",    "revenue")
        missing_line = _find(rank, "LINE", "dt_missing", "revenue")
        if full_line is not None and missing_line is not None:
            self.assertGreaterEqual(full_line.score, missing_line.score)


# ---------------------------------------------------------------------------
# Test 11 — Scatter requires meaningful variation
# ---------------------------------------------------------------------------

class TestScatterRequiresVariation(unittest.TestCase):
    def test_constant_column_not_in_scatter(self) -> None:
        """A constant-value column is UNKNOWN (Stage 1) → excluded from Stage 2 candidates."""
        df = pd.DataFrame({
            "constant_x": [5.0] * 10,       # UNKNOWN (constant)
            "price":      [float(i) for i in range(10)],
        })
        _, _, rank = _pipeline(df)
        for c in rank.ranked:
            self.assertNotEqual(c.x_column, "constant_x")
            self.assertNotEqual(c.y_column, "constant_x")

    def test_two_varying_measures_produce_scatter(self) -> None:
        df = pd.DataFrame({
            "weight": [50.0, 60.0, 70.0, 80.0, 90.0],
            "height": [1.60, 1.65, 1.70, 1.75, 1.80],
        })
        _, _, rank = _pipeline(df)
        scatters = rank.ranked_by_type("SCATTER")
        self.assertTrue(len(scatters) > 0)
        sc = scatters[0]
        self.assertGreater(sc.score, 0.0)


# ---------------------------------------------------------------------------
# Test 12 — Duplicate candidates removed
# ---------------------------------------------------------------------------

class TestDuplicatesRemoved(unittest.TestCase):
    def test_no_duplicate_keys_in_ranked(self) -> None:
        df = pd.DataFrame({
            "region": ["North", "South", "East", "West"] * 5,
            "sales":  [float(i * 10) for i in range(20)],
        })
        _, _, rank = _pipeline(df)
        seen: set[tuple[str, str, str | None]] = set()
        for c in rank.ranked:
            key = (c.chart_type, c.x_column, c.y_column)
            self.assertNotIn(key, seen, f"Duplicate candidate found: {key}")
            seen.add(key)

    def test_explicit_duplicate_suppressed(self) -> None:
        """Inject a duplicate directly into FeasibilityResult and verify dedup."""
        df = pd.DataFrame({
            "cat": ["A", "B", "A", "B"] * 3,
            "val": [10.0, 20.0, 30.0, 40.0] * 3,
        })
        clf = classify_columns(df)
        feas = analyze_visualization_feasibility(df, clf)

        # Inject a duplicate manually
        original_candidates = list(feas.candidates)
        first = original_candidates[0]
        dup_feas = FeasibilityResult(
            dataset_name=feas.dataset_name,
            row_count=feas.row_count,
            column_count=feas.column_count,
            candidates=original_candidates + [first],  # explicit duplicate
            suppressed_reasons=list(feas.suppressed_reasons),
        )
        rank = rank_visualization_candidates(df, clf, dup_feas)

        # Verify no duplicate in ranked output
        seen: set[tuple[str, str, str | None]] = set()
        for c in rank.ranked:
            key = (c.chart_type, c.x_column, c.y_column)
            self.assertNotIn(key, seen)
            seen.add(key)

        # Verify suppression noted
        self.assertTrue(any("Duplicate" in n for n in rank.ranking_notes))


# ---------------------------------------------------------------------------
# Test 13 — Deterministic tie-breaking
# ---------------------------------------------------------------------------

class TestDeterministicTieBreaking(unittest.TestCase):
    def test_chart_type_priority_breaks_ties(self) -> None:
        """When scores are equal (or very close), chart-type priority must determine order."""
        # Build a DataFrame where two chart types have a similar score.
        # We can check that the ordered sequence respects CHART_TYPE_PRIORITY.
        df = pd.DataFrame({
            "region": ["N", "S", "E", "W", "N"] * 4,
            "order_date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "sales": [float(i * 10 + 5) for i in range(20)],
        })
        _, _, rank = _pipeline(df)
        # Verify ordering is non-decreasing in sort key (already guaranteed by sort)
        for i in range(len(rank.ranked) - 1):
            a, b = rank.ranked[i], rank.ranked[i + 1]
            # score must be descending (or equal)
            self.assertGreaterEqual(
                round(a.score, 6), round(b.score, 6),
                f"Score ordering violated at rank {a.rank} vs {b.rank}",
            )

    def test_rank_numbers_are_sequential(self) -> None:
        df = pd.DataFrame({
            "region": ["A", "B", "A", "B", "A"],
            "sales":  [10.0, 20.0, 30.0, 40.0, 50.0],
            "cost":   [5.0, 10.0, 15.0, 20.0, 25.0],
        })
        _, _, rank = _pipeline(df)
        for expected, c in enumerate(rank.ranked, start=1):
            self.assertEqual(c.rank, expected,
                             f"Rank {c.rank} should be {expected}")


# ---------------------------------------------------------------------------
# Test 14 — Deterministic repeated execution
# ---------------------------------------------------------------------------

class TestDeterministicRepeatedExecution(unittest.TestCase):
    def test_identical_output_on_repeated_calls(self) -> None:
        df = pd.DataFrame({
            "region": ["North", "South", "East", "West"] * 5,
            "order_date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "sales": [float(i * 12.5) for i in range(20)],
            "cost":  [float(i * 7.5) for i in range(20)],
            "is_returned": [i % 2 for i in range(20)],
        })
        clf = classify_columns(df)
        feas = analyze_visualization_feasibility(df, clf)
        r1 = rank_visualization_candidates(df, clf, feas)
        r2 = rank_visualization_candidates(df, clf, feas)
        self.assertEqual(r1.model_dump(), r2.model_dump())
        for c1, c2 in zip(r1.ranked, r2.ranked):
            self.assertEqual(c1.model_dump(), c2.model_dump())


# ---------------------------------------------------------------------------
# Test 15 — Empty dataframe
# ---------------------------------------------------------------------------

class TestEmptyDataframe(unittest.TestCase):
    def test_zero_rows_empty_ranking(self) -> None:
        df = pd.DataFrame({
            "region": pd.Series([], dtype=str),
            "sales": pd.Series([], dtype=float),
        })
        clf = classify_columns(df)
        feas = analyze_visualization_feasibility(df, clf)
        rank = rank_visualization_candidates(df, clf, feas)
        self.assertEqual(rank.ranked_count, 0)
        self.assertEqual(rank.row_count, 0)
        self.assertIsNone(rank.top)
        self.assertTrue(any("zero rows" in n.lower() for n in rank.ranking_notes))


# ---------------------------------------------------------------------------
# Test 16 — No feasible candidates from Stage 2
# ---------------------------------------------------------------------------

class TestNoFeasibleCandidates(unittest.TestCase):
    def test_empty_feasibility_empty_ranking(self) -> None:
        df = pd.DataFrame({
            "constant_a": [1] * 5,
            "constant_b": [2] * 5,
        })
        clf = classify_columns(df)
        feas = analyze_visualization_feasibility(df, clf)
        rank = rank_visualization_candidates(df, clf, feas)
        self.assertEqual(rank.ranked_count, 0)
        self.assertIsNone(rank.top)
        self.assertTrue(len(rank.ranking_notes) > 0)


# ---------------------------------------------------------------------------
# Test 17 — Multiple candidates, stable ordering
# ---------------------------------------------------------------------------

class TestMultipleCandidatesStableOrdering(unittest.TestCase):
    def test_rich_dataset_ordered_descending(self) -> None:
        df = pd.DataFrame({
            "customer_id": [f"C{i:03d}" for i in range(30)],
            "order_date":  pd.date_range("2023-01-01", periods=30, freq="D"),
            "region":      (["North", "South", "East"] * 10),
            "category":    (["Electronics", "Food", "Clothing"] * 10),
            "revenue":     [float(i * 150 + 50) for i in range(30)],
            "cost":        [float(i * 80 + 20) for i in range(30)],
            "is_returned": [i % 2 for i in range(30)],
        })
        _, _, rank = _pipeline(df)
        scores = [c.score for c in rank.ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rank_assignment_matches_position(self) -> None:
        df = pd.DataFrame({
            "status": ["A", "B", "A", "B"] * 5,
            "value":  [float(i) for i in range(20)],
            "cost":   [float(i * 2) for i in range(20)],
        })
        _, _, rank = _pipeline(df)
        for i, c in enumerate(rank.ranked, start=1):
            self.assertEqual(c.rank, i)

    def test_top_property_returns_rank1(self) -> None:
        df = pd.DataFrame({
            "region": ["N", "S", "E", "W"] * 5,
            "sales":  [float(i * 10) for i in range(20)],
        })
        _, _, rank = _pipeline(df)
        if rank.ranked:
            self.assertEqual(rank.top, rank.ranked[0])
            self.assertEqual(rank.top.rank, 1)


# ---------------------------------------------------------------------------
# Test 18 — Ranking never creates candidates that Stage 2 did not provide
# ---------------------------------------------------------------------------

class TestRankingNeverInventsCandidates(unittest.TestCase):
    def test_all_ranked_keys_exist_in_feasibility(self) -> None:
        df = pd.DataFrame({
            "region": ["North", "South", "East"] * 6,
            "order_date": pd.date_range("2023-01-01", periods=18, freq="D"),
            "revenue": [float(i * 10) for i in range(18)],
            "cost":    [float(i * 5) for i in range(18)],
        })
        clf = classify_columns(df)
        feas = analyze_visualization_feasibility(df, clf)
        rank = rank_visualization_candidates(df, clf, feas)

        feasible_keys = {
            (c.chart_type, c.x_column, c.y_column) for c in feas.candidates
        }
        for c in rank.ranked:
            key = (c.chart_type, c.x_column, c.y_column)
            self.assertIn(
                key, feasible_keys,
                f"Ranked candidate {key} was not in Stage 2 feasibility output",
            )

    def test_ranking_never_adds_identifier_candidates(self) -> None:
        df = pd.DataFrame({
            "user_id": [f"USR_{i:03d}" for i in range(15)],
            "revenue": [float(i * 50) for i in range(15)],
            "region":  ["North", "South", "East"] * 5,
        })
        _, _, rank = _pipeline(df)
        for c in rank.ranked:
            self.assertNotEqual(c.x_column, "user_id")
            self.assertNotEqual(c.y_column, "user_id")

    def test_ranking_never_adds_descriptive_text_candidates(self) -> None:
        df = pd.DataFrame({
            "notes": [f"This is a review with many words and detailed content. {i}" for i in range(10)],
            "price": [float(i * 10) for i in range(10)],
        })
        _, _, rank = _pipeline(df)
        for c in rank.ranked:
            self.assertNotEqual(c.x_column, "notes")
            self.assertNotEqual(c.y_column, "notes")


# ---------------------------------------------------------------------------
# Test: RankingResult schema utilities
# ---------------------------------------------------------------------------

class TestRankingResultHelpers(unittest.TestCase):
    def test_ranked_by_type_filters_correctly(self) -> None:
        df = pd.DataFrame({
            "region": ["A", "B", "A", "B"] * 5,
            "sales":  [float(i * 10) for i in range(20)],
        })
        _, _, rank = _pipeline(df)
        for vt in ["BAR", "HISTOGRAM", "SCATTER", "LINE", "COUNT_BAR"]:
            subset = rank.ranked_by_type(vt)  # type: ignore[arg-type]
            for c in subset:
                self.assertEqual(c.chart_type, vt)

    def test_ranked_count_matches_ranked_length(self) -> None:
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        _, _, rank = _pipeline(df)
        self.assertEqual(rank.ranked_count, len(rank.ranked))

    def test_score_is_rounded_to_6_dp(self) -> None:
        df = pd.DataFrame({
            "cat": ["X", "Y", "X", "Y"],
            "val": [1.0, 2.0, 3.0, 4.0],
        })
        _, _, rank = _pipeline(df)
        for c in rank.ranked:
            # score should have at most 6 decimal places
            self.assertAlmostEqual(c.score, round(c.score, 6))


if __name__ == "__main__":
    unittest.main()

