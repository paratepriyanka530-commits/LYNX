"""Comprehensive tests for Pipeline 3 Stage 5 — Visualization Orchestrator."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from pipelines.pipeline3_results.visualization.orchestrator import (
    build_visualization_result,
)
from pipelines.pipeline3_results.visualization.schemas import (
    ChartData,
    ChartPoint,
    ColumnClassification,
    DatasetClassification,
    FeasibilityResult,
    RankedCandidate,
    RankingResult,
    VisualizationCandidate,
    VisualizationItem,
    VisualizationResult,
)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _small_realistic_df() -> pd.DataFrame:
    """Small realistic multi-column dataset (20 rows) for integration testing."""
    return pd.DataFrame({
        "order_id": [f"ORD_{i:03d}" for i in range(20)],
        "pizza_size": (["S", "M", "L", "XL"] * 5),
        "order_date": pd.date_range("2023-01-01", periods=20, freq="D"),
        "total_price": [float(i * 12.5 + 10.0) for i in range(20)],
        "quantity": [float((i % 4) + 1) for i in range(20)],
    })


def _mock_classification(columns: list[tuple[str, str]]) -> DatasetClassification:
    return DatasetClassification(
        dataset_name="test_ds",
        row_count=10,
        column_count=len(columns),
        columns=[
            ColumnClassification(
                name=name,
                role=role,  # type: ignore[arg-type]
                dtype="float64" if role == "MEASURE" else "object",
                unique_count=4,
                unique_ratio=0.4,
                missing_count=0,
                missing_ratio=0.0,
                signals=[],
            )
            for name, role in columns
        ],
        role_counts={r: sum(1 for _, role in columns if role == r) for r in ["MEASURE", "CATEGORY", "TEMPORAL", "IDENTIFIER", "DESCRIPTIVE_TEXT", "BOOLEAN", "UNKNOWN"]},
    )


def _mock_chart_data(has_data: bool = True, chart_type: str = "BAR", x: str = "cat", y: str | None = "meas") -> ChartData:
    return ChartData(
        chart_type=chart_type,  # type: ignore[arg-type]
        x_column=x,
        y_column=y,
        aggregation="sum" if y else "count",
        points=[ChartPoint(x="A", y=10.0)] if has_data else [],
        bins=[],
        signals=["mock chart data produced"] if has_data else ["unusable data"],
        usable_row_count=1 if has_data else 0,
        sampled=False,
        has_data=has_data,
    )


def _mock_ranked_candidate(rank: int, chart_type: str, x: str, y: str | None = None, score: float = 0.9) -> RankedCandidate:
    return RankedCandidate(
        chart_type=chart_type,  # type: ignore[arg-type]
        x_column=x,
        y_column=y,
        score=score,
        rank=rank,
        signals=[f"cand {rank} signals"],
    )


# ---------------------------------------------------------------------------
# Test 1 & 2 — Full Successful Pipeline & READY Status
# ---------------------------------------------------------------------------

class TestFullSuccessfulPipeline(unittest.TestCase):
    def test_full_pipeline_produces_ready_result(self) -> None:
        df = _small_realistic_df()
        result = build_visualization_result(df, dataset_name="pizza_sales")
        self.assertEqual(result.status, "READY")
        self.assertEqual(result.dataset_name, "pizza_sales")
        self.assertEqual(result.row_count, 20)
        self.assertEqual(result.column_count, 5)
        self.assertGreater(result.visualization_count, 0)
        self.assertLessEqual(result.visualization_count, 5)

        for viz in result.visualizations:
            self.assertIsInstance(viz, VisualizationItem)
            self.assertGreater(viz.rank, 0)
            self.assertTrue(viz.chart_data.has_data)
            self.assertGreaterEqual(viz.score, 0.0)
            self.assertLessEqual(viz.score, 1.0)
            self.assertGreater(len(viz.signals), 0)

    def test_ready_status_signals(self) -> None:
        df = _small_realistic_df()
        result = build_visualization_result(df)
        self.assertEqual(result.status, "READY")
        self.assertTrue(any("automatic visualizations produced" in s for s in result.signals))
        self.assertTrue(any("Stage 1 classified" in s for s in result.signals))
        self.assertTrue(any("Stage 2 produced" in s for s in result.signals))
        self.assertTrue(any("Stage 3 ranked" in s for s in result.signals))


# ---------------------------------------------------------------------------
# Test 3 — NO_MEANINGFUL_VISUALIZATION for empty dataframe
# ---------------------------------------------------------------------------

class TestEmptyDataframe(unittest.TestCase):
    def test_empty_dataframe_returns_no_meaningful_visualization(self) -> None:
        df = pd.DataFrame({"col_a": pd.Series([], dtype=float)})
        result = build_visualization_result(df, dataset_name="empty_test")
        self.assertEqual(result.status, "NO_MEANINGFUL_VISUALIZATION")
        self.assertEqual(result.row_count, 0)
        self.assertEqual(result.visualization_count, 0)
        self.assertIsNone(result.top)
        self.assertTrue(any("dataframe has zero rows" in s for s in result.signals))


# ---------------------------------------------------------------------------
# Test 4 — NO_MEANINGFUL_VISUALIZATION when no feasible candidates exist
# ---------------------------------------------------------------------------

class TestNoFeasibleCandidates(unittest.TestCase):
    def test_constant_only_dataframe_returns_no_meaningful_viz(self) -> None:
        # Constant columns are UNKNOWN -> Stage 2 produces 0 candidates
        df = pd.DataFrame({
            "c1": [42] * 10,
            "c2": ["constant"] * 10,
        })
        result = build_visualization_result(df, dataset_name="constant_ds")
        self.assertEqual(result.status, "NO_MEANINGFUL_VISUALIZATION")
        self.assertEqual(result.visualization_count, 0)
        self.assertTrue(any("no feasible visualization candidates" in s for s in result.signals))


# ---------------------------------------------------------------------------
# Test 5 — NO_MEANINGFUL_VISUALIZATION when all candidates fail Stage 4
# ---------------------------------------------------------------------------

class TestAllCandidatesFailStage4(unittest.TestCase):
    @patch("pipelines.pipeline3_results.visualization.orchestrator.build_chart_data")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.rank_visualization_candidates")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.analyze_visualization_feasibility")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.classify_columns")
    def test_all_stage4_unusable_returns_no_meaningful_viz(
        self, mock_cls, mock_feas, mock_rank, mock_build
    ) -> None:
        df = pd.DataFrame({"x": [1, 2, 3], "y": [3, 4, 5]})
        mock_cls.return_value = _mock_classification([("x", "MEASURE"), ("y", "MEASURE")])
        mock_feas.return_value = FeasibilityResult(
            row_count=3, column_count=2,
            candidates=[VisualizationCandidate(chart_type="HISTOGRAM", x_column="x")],
        )
        mock_rank.return_value = RankingResult(
            row_count=3, column_count=2,
            ranked=[_mock_ranked_candidate(1, "HISTOGRAM", "x")],
        )
        # Stage 4 returns has_data=False
        mock_build.return_value = _mock_chart_data(has_data=False, chart_type="HISTOGRAM", x="x", y=None)

        result = build_visualization_result(df)
        self.assertEqual(result.status, "NO_MEANINGFUL_VISUALIZATION")
        self.assertEqual(result.visualization_count, 0)
        self.assertTrue(any("none produced usable chart data" in s for s in result.signals))


# ---------------------------------------------------------------------------
# Test 6 — Top-ranked candidates are considered first
# ---------------------------------------------------------------------------

class TestTopRankedConsideredFirst(unittest.TestCase):
    @patch("pipelines.pipeline3_results.visualization.orchestrator.build_chart_data")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.rank_visualization_candidates")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.analyze_visualization_feasibility")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.classify_columns")
    def test_candidates_processed_in_stage3_order(
        self, mock_cls, mock_feas, mock_rank, mock_build
    ) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
        mock_cls.return_value = _mock_classification([("a", "MEASURE"), ("b", "MEASURE")])
        mock_feas.return_value = FeasibilityResult(row_count=3, column_count=2, candidates=[
            VisualizationCandidate(chart_type="HISTOGRAM", x_column="a"),
            VisualizationCandidate(chart_type="HISTOGRAM", x_column="b"),
        ])
        c1 = _mock_ranked_candidate(1, "HISTOGRAM", "a", score=0.95)
        c2 = _mock_ranked_candidate(2, "HISTOGRAM", "b", score=0.80)
        mock_rank.return_value = RankingResult(row_count=3, column_count=2, ranked=[c1, c2])

        processed_order = []
        def fake_build(dataframe, classification, cand):
            processed_order.append(cand.x_column)
            return _mock_chart_data(has_data=True, chart_type=cand.chart_type, x=cand.x_column)

        mock_build.side_effect = fake_build

        result = build_visualization_result(df, max_visualizations=2)
        self.assertEqual(processed_order, ["a", "b"])
        self.assertEqual(result.visualizations[0].x_column, "a")
        self.assertEqual(result.visualizations[1].x_column, "b")


# ---------------------------------------------------------------------------
# Test 7, 8, 9, 10 — max_visualizations=1, 3, 5/default, fewer usable
# ---------------------------------------------------------------------------

class TestMaxVisualizationsParameter(unittest.TestCase):
    def test_max_visualizations_1(self) -> None:
        df = _small_realistic_df()
        result = build_visualization_result(df, max_visualizations=1)
        self.assertEqual(result.status, "READY")
        self.assertEqual(result.visualization_count, 1)

    def test_max_visualizations_3(self) -> None:
        df = _small_realistic_df()
        result = build_visualization_result(df, max_visualizations=3)
        self.assertEqual(result.status, "READY")
        self.assertEqual(result.visualization_count, 3)

    def test_max_visualizations_default_is_5(self) -> None:
        df = _small_realistic_df()
        result = build_visualization_result(df)  # default max_visualizations=5
        self.assertEqual(result.status, "READY")
        self.assertLessEqual(result.visualization_count, 5)

    def test_fewer_usable_than_requested_maximum(self) -> None:
        # A simple DataFrame with only 1 category and 1 measure has few feasible candidates
        df = pd.DataFrame({
            "cat": ["A", "B", "A", "B"],
            "val": [10.0, 20.0, 30.0, 40.0],
        })
        # Request 100, should only return however many are actually usable without error
        result = build_visualization_result(df, max_visualizations=100)
        self.assertEqual(result.status, "READY")
        self.assertLess(result.visualization_count, 100)
        self.assertGreater(result.visualization_count, 0)


# ---------------------------------------------------------------------------
# Test 11 — Failed top candidate does not prevent lower candidates from being used
# ---------------------------------------------------------------------------

class TestFallbackOnTopCandidateFailure(unittest.TestCase):
    @patch("pipelines.pipeline3_results.visualization.orchestrator.build_chart_data")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.rank_visualization_candidates")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.analyze_visualization_feasibility")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.classify_columns")
    def test_continues_when_top_candidate_fails(
        self, mock_cls, mock_feas, mock_rank, mock_build
    ) -> None:
        df = pd.DataFrame({"x": [1, 2, 3], "y": [3, 4, 5]})
        mock_cls.return_value = _mock_classification([("x", "MEASURE"), ("y", "MEASURE")])
        mock_feas.return_value = FeasibilityResult(row_count=3, column_count=2, candidates=[
            VisualizationCandidate(chart_type="HISTOGRAM", x_column="x"),
            VisualizationCandidate(chart_type="HISTOGRAM", x_column="y"),
        ])
        c1 = _mock_ranked_candidate(1, "HISTOGRAM", "x", score=0.95)
        c2 = _mock_ranked_candidate(2, "HISTOGRAM", "y", score=0.85)
        mock_rank.return_value = RankingResult(row_count=3, column_count=2, ranked=[c1, c2])

        # Candidate 1 fails (unusable data), Candidate 2 succeeds
        def fake_build(dataframe, classification, cand):
            if cand.x_column == "x":
                return _mock_chart_data(has_data=False, chart_type=cand.chart_type, x="x")
            return _mock_chart_data(has_data=True, chart_type=cand.chart_type, x="y")

        mock_build.side_effect = fake_build

        result = build_visualization_result(df, max_visualizations=1)
        self.assertEqual(result.status, "READY")
        self.assertEqual(result.visualization_count, 1)
        self.assertEqual(result.visualizations[0].x_column, "y")
        self.assertEqual(result.visualizations[0].rank, 2)
        self.assertTrue(any("skipped" in s for s in result.signals))


# ---------------------------------------------------------------------------
# Test 12 — Deterministic repeated execution
# ---------------------------------------------------------------------------

class TestDeterministicRepeatedExecution(unittest.TestCase):
    def test_repeated_execution_identical_output(self) -> None:
        df = _small_realistic_df()
        r1 = build_visualization_result(df, dataset_name="pizza", max_visualizations=3)
        r2 = build_visualization_result(df, dataset_name="pizza", max_visualizations=3)
        self.assertEqual(r1.model_dump(), r2.model_dump())
        self.assertEqual(
            [v.model_dump() for v in r1.visualizations],
            [v.model_dump() for v in r2.visualizations],
        )


# ---------------------------------------------------------------------------
# Test 13 — Pipeline 1 report is passed into Stage 1
# ---------------------------------------------------------------------------

class TestPipeline1ReportPassed(unittest.TestCase):
    @patch("pipelines.pipeline3_results.visualization.orchestrator.classify_columns")
    def test_pipeline1_report_forwarded_to_classify_columns(self, mock_cls) -> None:
        df = pd.DataFrame({"col": [1, 2, 3]})
        fake_report = {"semantic_profile": {"columns": []}}
        mock_cls.return_value = _mock_classification([("col", "MEASURE")])

        build_visualization_result(df, pipeline1_report=fake_report)
        mock_cls.assert_called_once_with(
            dataframe=df,
            pipeline1_report=fake_report,
            dataset_name=None,
        )


# ---------------------------------------------------------------------------
# Test 14 — Dataset name is preserved
# ---------------------------------------------------------------------------

class TestDatasetNamePreserved(unittest.TestCase):
    def test_dataset_name_in_result(self) -> None:
        df = _small_realistic_df()
        result = build_visualization_result(df, dataset_name="my_custom_dataset")
        self.assertEqual(result.dataset_name, "my_custom_dataset")

    def test_none_dataset_name_handled(self) -> None:
        df = _small_realistic_df()
        result = build_visualization_result(df, dataset_name=None)
        self.assertIsNone(result.dataset_name)


# ---------------------------------------------------------------------------
# Test 15 — Invalid max_visualizations is rejected
# ---------------------------------------------------------------------------

class TestInvalidMaxVisualizations(unittest.TestCase):
    def test_zero_rejected(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        with self.assertRaises(ValueError):
            build_visualization_result(df, max_visualizations=0)

    def test_negative_rejected(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        with self.assertRaises(ValueError):
            build_visualization_result(df, max_visualizations=-5)

    def test_non_int_rejected(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        with self.assertRaises(ValueError):
            build_visualization_result(df, max_visualizations="3")  # type: ignore[arg-type]

    def test_boolean_rejected(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        with self.assertRaises(ValueError):
            build_visualization_result(df, max_visualizations=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 16 — No duplicate visualizations
# ---------------------------------------------------------------------------

class TestNoDuplicateVisualizations(unittest.TestCase):
    def test_no_duplicate_chart_structures(self) -> None:
        df = _small_realistic_df()
        result = build_visualization_result(df, max_visualizations=10)
        seen = set()
        for viz in result.visualizations:
            key = (viz.chart_type, viz.x_column, viz.y_column)
            self.assertNotIn(key, seen, f"Duplicate visualization found: {key}")
            seen.add(key)


# ---------------------------------------------------------------------------
# Test 17 — Visualization rank follows Stage 3 ranking
# ---------------------------------------------------------------------------

class TestVisualizationRankFollowsStage3(unittest.TestCase):
    def test_rank_numbers_match_stage3(self) -> None:
        df = _small_realistic_df()
        result = build_visualization_result(df, max_visualizations=5)
        for i, viz in enumerate(result.visualizations):
            # Rank must be >= 1 and strictly increasing
            self.assertGreaterEqual(viz.rank, 1)
            if i > 0:
                self.assertGreater(viz.rank, result.visualizations[i - 1].rank)


# ---------------------------------------------------------------------------
# Test 18 — Stage 5 does not invent candidates
# ---------------------------------------------------------------------------

class TestStage5DoesNotInventCandidates(unittest.TestCase):
    def test_all_visualizations_originate_from_stage2_feasibility(self) -> None:
        df = _small_realistic_df()
        from pipelines.pipeline3_results.visualization.column_classifier import classify_columns
        from pipelines.pipeline3_results.visualization.feasibility import analyze_visualization_feasibility

        clf = classify_columns(df)
        feas = analyze_visualization_feasibility(df, clf)
        feasible_set = {(c.chart_type, c.x_column, c.y_column) for c in feas.candidates}

        result = build_visualization_result(df, max_visualizations=10)
        for viz in result.visualizations:
            key = (viz.chart_type, viz.x_column, viz.y_column)
            self.assertIn(key, feasible_set, f"Stage 5 invented candidate: {key}")


# ---------------------------------------------------------------------------
# Test 19 — Stage 5 does not invent chart data
# ---------------------------------------------------------------------------

class TestStage5DoesNotInventChartData(unittest.TestCase):
    def test_chart_data_matches_stage4_output(self) -> None:
        df = _small_realistic_df()
        from pipelines.pipeline3_results.visualization.column_classifier import classify_columns
        from pipelines.pipeline3_results.visualization.data_builder import build_chart_data
        from pipelines.pipeline3_results.visualization.feasibility import analyze_visualization_feasibility
        from pipelines.pipeline3_results.visualization.ranker import rank_visualization_candidates

        clf = classify_columns(df)
        feas = analyze_visualization_feasibility(df, clf)
        ranking = rank_visualization_candidates(df, clf, feas)

        result = build_visualization_result(df, max_visualizations=3)
        for viz in result.visualizations:
            cand = next(c for c in ranking.ranked if c.rank == viz.rank)
            direct_data = build_chart_data(df, clf, cand)
            self.assertEqual(viz.chart_data.model_dump(), direct_data.model_dump())


# ---------------------------------------------------------------------------
# Test 20 — Candidate-level failure is recorded and pipeline continues
# ---------------------------------------------------------------------------

class TestCandidateLevelExceptionRecorded(unittest.TestCase):
    @patch("pipelines.pipeline3_results.visualization.orchestrator.build_chart_data")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.rank_visualization_candidates")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.analyze_visualization_feasibility")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.classify_columns")
    def test_exception_in_build_chart_data_handled_gracefully(
        self, mock_cls, mock_feas, mock_rank, mock_build
    ) -> None:
        df = pd.DataFrame({"x": [1, 2, 3], "y": [3, 4, 5]})
        mock_cls.return_value = _mock_classification([("x", "MEASURE"), ("y", "MEASURE")])
        mock_feas.return_value = FeasibilityResult(row_count=3, column_count=2, candidates=[
            VisualizationCandidate(chart_type="HISTOGRAM", x_column="x"),
            VisualizationCandidate(chart_type="HISTOGRAM", x_column="y"),
        ])
        c1 = _mock_ranked_candidate(1, "HISTOGRAM", "x", score=0.95)
        c2 = _mock_ranked_candidate(2, "HISTOGRAM", "y", score=0.85)
        mock_rank.return_value = RankingResult(row_count=3, column_count=2, ranked=[c1, c2])

        def fake_build(dataframe, classification, cand):
            if cand.x_column == "x":
                raise RuntimeError("Synthetic failure in chart data building")
            return _mock_chart_data(has_data=True, chart_type=cand.chart_type, x="y")

        mock_build.side_effect = fake_build

        result = build_visualization_result(df, max_visualizations=1)
        self.assertEqual(result.status, "READY")
        self.assertEqual(result.visualization_count, 1)
        self.assertEqual(result.visualizations[0].x_column, "y")
        # Check signal notes the exception
        self.assertTrue(any("failed data generation" in s and "RuntimeError" in s for s in result.signals))


# ---------------------------------------------------------------------------
# Test 21 — Output is JSON serializable
# ---------------------------------------------------------------------------

class TestJsonSerializable(unittest.TestCase):
    def test_full_result_is_json_serializable(self) -> None:
        df = _small_realistic_df()
        result = build_visualization_result(df, dataset_name="pizza_sales", max_visualizations=5)
        dumped = result.model_dump()
        raw_json = json.dumps(dumped)
        self.assertIsInstance(raw_json, str)
        reloaded = json.loads(raw_json)
        self.assertEqual(reloaded["status"], "READY")
        self.assertEqual(len(reloaded["visualizations"]), result.visualization_count)

    def test_no_meaningful_viz_is_json_serializable(self) -> None:
        df = pd.DataFrame({"empty": pd.Series([], dtype=float)})
        result = build_visualization_result(df, dataset_name="empty")
        dumped = result.model_dump()
        raw_json = json.dumps(dumped)
        self.assertIsInstance(raw_json, str)
        reloaded = json.loads(raw_json)
        self.assertEqual(reloaded["status"], "NO_MEANINGFUL_VISUALIZATION")


# ---------------------------------------------------------------------------
# Test 22 — Small Dataset Guard (<= 2 rows)
# ---------------------------------------------------------------------------

class TestSmallDatasetGuard(unittest.TestCase):
    def test_one_row_dataframe_returns_no_meaningful_visualization(self) -> None:
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        result = build_visualization_result(df, dataset_name="single_row")
        self.assertEqual(result.status, "NO_MEANINGFUL_VISUALIZATION")
        self.assertEqual(result.row_count, 1)
        self.assertEqual(result.visualization_count, 0)
        self.assertTrue(
            any("insufficient rows for meaningful visualization (minimum 3 rows required)" in s for s in result.signals)
        )

    def test_two_row_dataframe_returns_no_meaningful_visualization(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = build_visualization_result(df, dataset_name="two_rows")
        self.assertEqual(result.status, "NO_MEANINGFUL_VISUALIZATION")
        self.assertEqual(result.row_count, 2)
        self.assertEqual(result.visualization_count, 0)
        self.assertTrue(
            any("insufficient rows for meaningful visualization (minimum 3 rows required)" in s for s in result.signals)
        )


# ---------------------------------------------------------------------------
# Test 23 — Intent Diversity & Candidate Selection
# ---------------------------------------------------------------------------

class TestIntentDiversity(unittest.TestCase):
    @patch("pipelines.pipeline3_results.visualization.orchestrator.build_chart_data")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.rank_visualization_candidates")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.analyze_visualization_feasibility")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.classify_columns")
    def test_intent_diversity_prevents_count_bar_domination(
        self, mock_cls, mock_feas, mock_rank, mock_build
    ) -> None:
        """Diverse intents (COMPARISON, TREND, RELATIONSHIP) are selected even if multiple COUNT_BARs have higher scores."""
        df = pd.DataFrame({
            "cat": ["A", "B", "C"],
            "num": [1.0, 2.0, 3.0],
            "dt": ["2023-01-01", "2023-01-02", "2023-01-03"],
        })
        mock_cls.return_value = _mock_classification([("cat", "CATEGORY"), ("num", "MEASURE"), ("dt", "TEMPORAL")])
        mock_feas.return_value = FeasibilityResult(row_count=3, column_count=3, candidates=[])

        c1 = _mock_ranked_candidate(1, "COUNT_BAR", "cat1", score=0.99)
        c2 = _mock_ranked_candidate(2, "COUNT_BAR", "cat2", score=0.98)
        c3 = _mock_ranked_candidate(3, "COUNT_BAR", "cat3", score=0.97)
        c4 = _mock_ranked_candidate(4, "BAR", "cat1", "num", score=0.90)
        c5 = _mock_ranked_candidate(5, "LINE", "dt", "num", score=0.85)
        c6 = _mock_ranked_candidate(6, "SCATTER", "num", "num2", score=0.80)
        cands = [c1, c2, c3, c4, c5, c6]
        mock_feas.return_value = FeasibilityResult(
            row_count=3,
            column_count=3,
            candidates=[VisualizationCandidate(chart_type=c.chart_type, x_column=c.x_column, y_column=c.y_column) for c in cands],
        )
        mock_rank.return_value = RankingResult(row_count=3, column_count=3, ranked=cands)

        mock_build.side_effect = lambda df, cls, cand: _mock_chart_data(
            has_data=True, chart_type=cand.chart_type, x=cand.x_column, y=cand.y_column
        )

        result = build_visualization_result(df, max_visualizations=4)
        self.assertEqual(result.status, "READY")
        self.assertEqual(result.visualization_count, 4)

        chart_types = [v.chart_type for v in result.visualizations]
        self.assertIn("COUNT_BAR", chart_types)
        self.assertIn("BAR", chart_types)
        self.assertIn("LINE", chart_types)
        self.assertIn("SCATTER", chart_types)
        # c2 and c3 must not have crowded out LINE and SCATTER
        self.assertEqual(chart_types.count("COUNT_BAR"), 1)

    @patch("pipelines.pipeline3_results.visualization.orchestrator.build_chart_data")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.rank_visualization_candidates")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.analyze_visualization_feasibility")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.classify_columns")
    def test_missing_intent_not_forced(
        self, mock_cls, mock_feas, mock_rank, mock_build
    ) -> None:
        """When an intent (e.g., TREND) has no feasible candidates, it is cleanly omitted without error."""
        df = pd.DataFrame({"cat": ["A", "B", "C"], "num": [1.0, 2.0, 3.0]})
        mock_cls.return_value = _mock_classification([("cat", "CATEGORY"), ("num", "MEASURE")])

        c1 = _mock_ranked_candidate(1, "COUNT_BAR", "cat", score=0.99)
        c2 = _mock_ranked_candidate(2, "BAR", "cat", "num", score=0.90)
        cands = [c1, c2]
        mock_feas.return_value = FeasibilityResult(
            row_count=3,
            column_count=2,
            candidates=[VisualizationCandidate(chart_type=c.chart_type, x_column=c.x_column, y_column=c.y_column) for c in cands],
        )
        mock_rank.return_value = RankingResult(row_count=3, column_count=2, ranked=cands)

        mock_build.side_effect = lambda df, cls, cand: _mock_chart_data(
            has_data=True, chart_type=cand.chart_type, x=cand.x_column, y=cand.y_column
        )

        result = build_visualization_result(df, max_visualizations=5)
        self.assertEqual(result.status, "READY")
        self.assertEqual(result.visualization_count, 2)
        chart_types = [v.chart_type for v in result.visualizations]
        self.assertNotIn("LINE", chart_types)

    @patch("pipelines.pipeline3_results.visualization.orchestrator.build_chart_data")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.rank_visualization_candidates")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.analyze_visualization_feasibility")
    @patch("pipelines.pipeline3_results.visualization.orchestrator.classify_columns")
    def test_redundant_same_chart_and_dimension_avoided(
        self, mock_cls, mock_feas, mock_rank, mock_build
    ) -> None:
        """Redundant candidates sharing the same chart type and primary dimension are deprioritized."""
        df = pd.DataFrame({"cat": ["A", "B", "C"], "m1": [1.0, 2.0, 3.0], "m2": [4.0, 5.0, 6.0]})
        mock_cls.return_value = _mock_classification([("cat", "CATEGORY"), ("m1", "MEASURE"), ("m2", "MEASURE")])

        c1 = _mock_ranked_candidate(1, "BAR", "cat", "m1", score=0.95)
        c2 = _mock_ranked_candidate(2, "BAR", "cat", "m2", score=0.94)  # redundant BAR on same x_column
        c3 = _mock_ranked_candidate(3, "HISTOGRAM", "m1", score=0.85)
        cands = [c1, c2, c3]
        mock_feas.return_value = FeasibilityResult(
            row_count=3,
            column_count=3,
            candidates=[VisualizationCandidate(chart_type=c.chart_type, x_column=c.x_column, y_column=c.y_column) for c in cands],
        )
        mock_rank.return_value = RankingResult(row_count=3, column_count=3, ranked=cands)

        mock_build.side_effect = lambda df, cls, cand: _mock_chart_data(
            has_data=True, chart_type=cand.chart_type, x=cand.x_column, y=cand.y_column
        )

        result = build_visualization_result(df, max_visualizations=2)
        self.assertEqual(result.visualization_count, 2)
        keys = [(v.chart_type, v.x_column, v.y_column) for v in result.visualizations]
        self.assertIn(("BAR", "cat", "m1"), keys)
        self.assertIn(("HISTOGRAM", "m1", None), keys)
        self.assertNotIn(("BAR", "cat", "m2"), keys)


# ---------------------------------------------------------------------------
# Test Type Guard on DataFrame
# ---------------------------------------------------------------------------

class TestTypeGuard(unittest.TestCase):
    def test_non_dataframe_raises_typeerror(self) -> None:
        with self.assertRaises(TypeError):
            build_visualization_result([1, 2, 3])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

