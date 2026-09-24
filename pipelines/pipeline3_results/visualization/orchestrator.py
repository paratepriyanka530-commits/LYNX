"""Deterministic Visualization Orchestrator — Pipeline 3, Stage 5.

Stage 5 of Pipeline 3: composes Stages 1–4 into a single, cohesive,
deterministic automatic visualization engine.

Given a prepared pandas DataFrame (and optional Pipeline 1 evidence),
this orchestrator answers:
    "Which meaningful visualizations can LYNX automatically present?"

Pipeline Composition Order:
    Stage 1: Column Intelligence (classify_columns)
        ↓
    Stage 2: Visualization Feasibility (analyze_visualization_feasibility)
        ↓
    Stage 3: Visualization Candidate Ranking (rank_visualization_candidates)
        ↓
    Stage 4: Deterministic Chart Data Generation (build_chart_data)
        ↓
    Stage 5: Filter & Assembly (build_visualization_result)

Design Principles:
* Fully deterministic: identical inputs yield identical outputs.
* No chart rendering, no UI code, no external service calls, no LLMs.
* Tolerant of candidate-level data failures: if a top candidate fails Stage 4,
  it continues evaluating subsequent ranked candidates until `max_visualizations`
  usable visualizations are collected or all candidates are exhausted.
* Returns exactly two high-level statuses:
  - "READY"
  - "NO_MEANINGFUL_VISUALIZATION"
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from pipelines.pipeline3_results.visualization.column_classifier import (
    classify_columns,
)
from pipelines.pipeline3_results.visualization.data_builder import (
    build_chart_data,
)
from pipelines.pipeline3_results.visualization.feasibility import (
    analyze_visualization_feasibility,
)
from pipelines.pipeline3_results.visualization.ranker import (
    rank_visualization_candidates,
)
from pipelines.pipeline3_results.visualization.schemas import (
    VisualizationItem,
    VisualizationResult,
    VisualizationStatus,
)


def build_visualization_result(
    dataframe: pd.DataFrame,
    pipeline1_report: Mapping[str, Any] | None = None,
    dataset_name: str | None = None,
    max_visualizations: int = 5,
) -> VisualizationResult:
    """Build a complete, deterministic automatic visualization result for a dataset.

    Composes Stage 1 (column intelligence), Stage 2 (feasibility analysis),
    Stage 3 (candidate ranking), and Stage 4 (chart data generation) into a
    single public call.

    Parameters
    ----------
    dataframe:
        The cleaned/prepared pandas DataFrame to visualize.
    pipeline1_report:
        Optional Pipeline 1 cleaning / profiling report mapping.
    dataset_name:
        Optional human-readable dataset name identifier.
    max_visualizations:
        Maximum number of usable visualizations to return (default: 5).
        Must be a positive integer (> 0).

    Returns
    -------
    VisualizationResult
        Deterministic result with status "READY" or "NO_MEANINGFUL_VISUALIZATION",
        containing ordered `VisualizationItem` entries and factual audit signals.

    Raises
    ------
    ValueError
        If `max_visualizations` is not a positive integer.
    TypeError
        If `dataframe` is not a pandas DataFrame.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"dataframe must be a pandas DataFrame, got {type(dataframe).__name__}")

    if isinstance(max_visualizations, bool) or not isinstance(max_visualizations, int) or max_visualizations <= 0:
        raise ValueError(
            f"max_visualizations must be a positive integer (> 0), got {max_visualizations!r}"
        )

    row_count = len(dataframe)
    column_count = len(dataframe.columns)
    signals: list[str] = []

    # -----------------------------------------------------------------------
    # Guard: empty or insufficient rows (<= 2 rows)
    # -----------------------------------------------------------------------
    if row_count == 0:
        signals.append("dataframe has zero rows")
        signals.append("no feasible visualization candidates")
        return VisualizationResult(
            dataset_name=dataset_name,
            row_count=0,
            column_count=column_count,
            status="NO_MEANINGFUL_VISUALIZATION",
            visualizations=[],
            signals=signals,
        )

    if row_count <= 2:
        signals.append(
            f"dataframe has {row_count} rows — insufficient rows for meaningful visualization (minimum 3 rows required)"
        )
        signals.append("no feasible visualization candidates")
        return VisualizationResult(
            dataset_name=dataset_name,
            row_count=row_count,
            column_count=column_count,
            status="NO_MEANINGFUL_VISUALIZATION",
            visualizations=[],
            signals=signals,
        )

    # -----------------------------------------------------------------------
    # Stage 1: Column Intelligence
    # -----------------------------------------------------------------------
    classification = classify_columns(
        dataframe=dataframe,
        pipeline1_report=pipeline1_report,
        dataset_name=dataset_name,
    )
    signals.append(f"Stage 1 classified {classification.column_count} columns")

    # -----------------------------------------------------------------------
    # Stage 2: Visualization Feasibility
    # -----------------------------------------------------------------------
    feasibility = analyze_visualization_feasibility(
        dataframe=dataframe,
        classification=classification,
    )
    signals.append(f"Stage 2 produced {feasibility.candidate_count} feasible candidates")

    if feasibility.candidate_count == 0:
        signals.append("no feasible visualization candidates")
        return VisualizationResult(
            dataset_name=dataset_name,
            row_count=row_count,
            column_count=column_count,
            status="NO_MEANINGFUL_VISUALIZATION",
            visualizations=[],
            signals=signals,
        )

    # -----------------------------------------------------------------------
    # Stage 3: Candidate Ranking
    # -----------------------------------------------------------------------
    ranking = rank_visualization_candidates(
        dataframe=dataframe,
        classification=classification,
        feasibility_result=feasibility,
    )
    signals.append(f"Stage 3 ranked {ranking.ranked_count} candidates")

    if ranking.ranked_count == 0:
        signals.append("no ranked visualization candidates")
        return VisualizationResult(
            dataset_name=dataset_name,
            row_count=row_count,
            column_count=column_count,
            status="NO_MEANINGFUL_VISUALIZATION",
            visualizations=[],
            signals=signals,
        )

    # -----------------------------------------------------------------------
    # Stage 4 & 5: Intent-Aware Diversity Selection & Chart Data Generation
    # -----------------------------------------------------------------------
    _CHART_INTENTS: dict[str, str] = {
        "COUNT_BAR": "DISTRIBUTION",
        "HISTOGRAM": "DISTRIBUTION",
        "BAR": "COMPARISON",
        "LINE": "TREND",
        "SCATTER": "RELATIONSHIP",
    }

    chart_cache: dict[tuple[str, str, str | None], ChartData] = {}
    skipped_cands: set[tuple[str, str, str | None]] = set()
    skipped_count = 0

    def _get_chart_data(cand: RankedCandidate) -> ChartData | None:
        nonlocal skipped_count
        key = (cand.chart_type, cand.x_column, cand.y_column)
        if key in chart_cache:
            return chart_cache[key]
        if key in skipped_cands:
            return None

        try:
            cd = build_chart_data(dataframe, classification, cand)
        except Exception as exc:
            skipped_count += 1
            skipped_cands.add(key)
            signals.append(
                f"candidate ({cand.chart_type}, {cand.x_column!r}, {cand.y_column!r}) "
                f"failed data generation: {type(exc).__name__}: {exc}"
            )
            return None

        if not cd.has_data:
            skipped_count += 1
            skipped_cands.add(key)
            signals.append(
                f"candidate ({cand.chart_type}, {cand.x_column!r}, {cand.y_column!r}) "
                f"skipped: chart data unusable"
            )
            return None

        chart_cache[key] = cd
        return cd

    def _is_redundant(
        cand: RankedCandidate,
        chosen: list[tuple[RankedCandidate, ChartData]],
        strict_dimension: bool,
    ) -> bool:
        for s_cand, _ in chosen:
            # Exact same candidate
            if (
                cand.chart_type == s_cand.chart_type
                and cand.x_column == s_cand.x_column
                and cand.y_column == s_cand.y_column
            ):
                return True

            if strict_dimension:
                # Same 1D chart type on same column
                if (
                    cand.chart_type in ("COUNT_BAR", "HISTOGRAM")
                    and s_cand.chart_type == cand.chart_type
                    and cand.x_column == s_cand.x_column
                ):
                    return True

                # Same BAR chart on same category dimension
                if (
                    cand.chart_type == "BAR"
                    and s_cand.chart_type == "BAR"
                    and cand.x_column == s_cand.x_column
                ):
                    return True

                # Same SCATTER pair regardless of order
                if (
                    cand.chart_type == "SCATTER"
                    and s_cand.chart_type == "SCATTER"
                    and {cand.x_column, cand.y_column} == {s_cand.x_column, s_cand.y_column}
                ):
                    return True

                # Same LINE on same temporal column and measure
                if (
                    cand.chart_type == "LINE"
                    and s_cand.chart_type == "LINE"
                    and cand.x_column == s_cand.x_column
                    and cand.y_column == s_cand.y_column
                ):
                    return True

        return False

    selected_pairs: list[tuple[RankedCandidate, ChartData]] = []

    # Group candidates by analytical intent preserving Stage 3 order
    intent_candidates: dict[str, list[RankedCandidate]] = {
        "DISTRIBUTION": [],
        "COMPARISON": [],
        "TREND": [],
        "RELATIONSHIP": [],
    }
    for cand in ranking.ranked:
        intent = _CHART_INTENTS.get(cand.chart_type, "DISTRIBUTION")
        intent_candidates[intent].append(cand)

    # Order available intents by the highest score of their top candidate descending
    available_intents = [
        intent
        for intent, cands in intent_candidates.items()
        if len(cands) > 0
    ]
    available_intents.sort(
        key=lambda it: intent_candidates[it][0].score if intent_candidates[it] else 0.0,
        reverse=True,
    )

    # Pass 1: Intent Diversity Pass (give each available intent an opportunity to contribute)
    for intent in available_intents:
        if len(selected_pairs) >= max_visualizations:
            break
        for cand in intent_candidates[intent]:
            if _is_redundant(cand, selected_pairs, strict_dimension=True):
                continue
            cd = _get_chart_data(cand)
            if cd is not None and cd.has_data:
                selected_pairs.append((cand, cd))
                break

    # Pass 2: Score Fill with Dimension Diversity Check
    if len(selected_pairs) < max_visualizations:
        for cand in ranking.ranked:
            if len(selected_pairs) >= max_visualizations:
                break
            if _is_redundant(cand, selected_pairs, strict_dimension=True):
                continue
            cd = _get_chart_data(cand)
            if cd is not None and cd.has_data:
                selected_pairs.append((cand, cd))

    # Pass 3: Fallback Fill without Strict Dimension Check (only skip exact duplicates)
    if len(selected_pairs) < max_visualizations:
        for cand in ranking.ranked:
            if len(selected_pairs) >= max_visualizations:
                break
            if _is_redundant(cand, selected_pairs, strict_dimension=False):
                continue
            cd = _get_chart_data(cand)
            if cd is not None and cd.has_data:
                selected_pairs.append((cand, cd))

    # Sort selected visualizations strictly by their Stage 3 rank
    selected_pairs.sort(key=lambda pair: pair[0].rank)

    visualizations: list[VisualizationItem] = []
    for cand, chart_data in selected_pairs:
        item_signals = list(cand.signals)
        for s in chart_data.signals:
            if s not in item_signals:
                item_signals.append(s)

        visualizations.append(
            VisualizationItem(
                rank=cand.rank,
                chart_type=cand.chart_type,
                x_column=cand.x_column,
                y_column=cand.y_column,
                score=cand.score,
                chart_data=chart_data,
                signals=item_signals,
            )
        )

    # -----------------------------------------------------------------------
    # Stage 5: Final Status & Result Assembly
    # -----------------------------------------------------------------------
    if visualizations:
        status: VisualizationStatus = "READY"
        signals.append(f"{len(visualizations)} automatic visualizations produced")
        intents_covered = sorted({_CHART_INTENTS.get(v.chart_type, "DISTRIBUTION") for v in visualizations})
        signals.append(f"analytical intents covered: {', '.join(intents_covered)}")
        if skipped_count > 0:
            signals.append(f"{skipped_count} candidates were skipped because chart data was unusable")
    else:
        status = "NO_MEANINGFUL_VISUALIZATION"
        signals.append("ranked candidates existed but none produced usable chart data")

    return VisualizationResult(
        dataset_name=dataset_name,
        row_count=row_count,
        column_count=column_count,
        status=status,
        visualizations=visualizations,
        signals=signals,
    )


