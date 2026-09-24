"""Deterministic Visualization Candidate Ranker — Pipeline 3, Stage 3.

Stage 3 of Pipeline 3: given a cleaned pandas DataFrame, the Stage 1
``DatasetClassification``, and the Stage 2 ``FeasibilityResult``, produce
an **ordered** list of visualization candidates ranked by a deterministic,
fully documented usefulness score.

This module answers:
    "Among the feasible visualizations, which ones are most meaningful
    for automatic presentation?"

It does NOT:
  - reclassify columns (Stage 1's job)
  - rediscover feasibility (Stage 2's job)
  - render charts or produce chart data
  - invoke any ML model, NLP, LLM, RAG, embeddings, or external service

Design principles
-----------------
* Consumes Stage 2 ``FeasibilityResult`` exclusively — never re-runs Stage 2.
* Every score component is a normalized float in [0.0, 1.0].
* All weights are module-level named constants with documented rationale.
* Tie-breaking is fully deterministic: score → chart-type priority → x_column
  name → y_column name (None sorts before any string).
* Duplicates (same chart_type, x_column, y_column triple) are deduplicated
  before ranking.
* Repeated calls with identical inputs produce byte-for-byte identical output.

Scoring formula
---------------
Each candidate's score is a weighted sum of normalized components:

    score = w_coverage    * coverage_component
          + w_variation   * variation_component
          + w_cardinality * cardinality_component
          + w_temporal    * temporal_component    (LINE only)
          + w_pair        * pair_component        (SCATTER only)

All weights sum to 1.0 for each chart type (see per-type weight tables below).
Scores are clamped to [0.0, 1.0] to guard against floating-point edge cases.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from pipelines.pipeline3_results.visualization.schemas import (
    CHART_TYPE_PRIORITY,
    DatasetClassification,
    FeasibilityResult,
    RankedCandidate,
    RankingResult,
    VisualizationCandidate,
)

# ---------------------------------------------------------------------------
# Scoring weight tables (all weights sum to 1.0 within each chart type)
# ---------------------------------------------------------------------------
# Rationale for weighting choices:
#   coverage    — columns with mostly-missing data produce misleading charts.
#   variation   — constant columns are useless for visualization.
#   cardinality — for categorical dimensions, readable cardinality matters most.
#   temporal    — for LINE charts, spread across time drives usefulness.
#   pair        — for SCATTER, both axes must be independently usable.

# BAR: aggregated measure across a category dimension.
#   cardinality (dimension readability) weighted highest,
#   then measure coverage and variation.
_W_BAR = {
    "coverage_x":     0.20,   # category non-missing coverage
    "cardinality_x":  0.35,   # category cardinality fitness
    "repetition_x":   0.15,   # category observations are repeated
    "coverage_y":     0.15,   # measure non-missing coverage
    "variation_y":    0.15,   # measure has meaningful variation
}

# LINE: measure over time — temporal continuity is paramount.
_W_LINE = {
    "coverage_x":     0.15,   # temporal non-missing coverage
    "temporal_x":     0.35,   # distinct usable temporal periods
    "coverage_y":     0.20,   # measure non-missing coverage
    "variation_y":    0.30,   # measure variation over time
}

# SCATTER: two varying measures — coverage and variation of both axes matter.
_W_SCATTER = {
    "coverage_x":     0.20,
    "variation_x":    0.30,
    "coverage_y":     0.20,
    "variation_y":    0.30,
}

# HISTOGRAM: single measure distribution — variation and coverage.
_W_HISTOGRAM = {
    "coverage_x":     0.40,
    "variation_x":    0.60,
}

# COUNT_BAR: category/boolean frequency — cardinality and repetition.
_W_COUNT_BAR = {
    "coverage_x":     0.25,
    "cardinality_x":  0.40,
    "repetition_x":   0.35,
}

# Ideal cardinality range for BAR / COUNT_BAR chart dimensions.
# Rationale: 3–15 distinct values produce readable bar charts.
_IDEAL_CARDINALITY_LOW: int = 3
_IDEAL_CARDINALITY_HIGH: int = 15
# Hard upper limit (same as Stage 2's _MAX_CATEGORY_CARDINALITY = 50).
_MAX_CARDINALITY: int = 50

# A measure is considered "constant" (zero variation) if its coefficient of
# variation (std / |mean|) is below this threshold.  Documented here so that
# tests can import it for verification.
_CONSTANT_CV_THRESHOLD: float = 1e-6

# Minimum useful unique temporal periods for a LINE chart.
_MIN_TEMPORAL_PERIODS: int = 3


# ---------------------------------------------------------------------------
# Per-series signal measurement helpers
# ---------------------------------------------------------------------------

def _coverage(series: pd.Series) -> tuple[float, str]:
    """Return (non-null fraction in [0,1], signal string)."""
    n = len(series)
    if n == 0:
        return 0.0, "coverage=0.00 (empty column)"
    non_null = int(series.notna().sum())
    cov = round(non_null / n, 4)
    return cov, f"coverage={cov:.0%} ({non_null}/{n} non-missing values)"


def _numeric_variation(series: pd.Series) -> tuple[float, str]:
    """Return (variation score in [0,1], signal string).

    Uses a sigmoid-like transform of the coefficient of variation so that
    the score is always in [0, 1]:
        variation_score = 1 - exp(-CV)
    A completely constant column → CV=0 → score=0.
    A column with CV >= ~2.3 → score >= 0.9.
    """
    non_null = series.dropna()
    if not pd.api.types.is_numeric_dtype(non_null.dtype):
        coerced = pd.to_numeric(non_null, errors="coerce").dropna()
    else:
        coerced = non_null[~non_null.isin([float("inf"), float("-inf")])]

    if len(coerced) < 2:
        return 0.0, "variation=0.00 (fewer than 2 usable values)"

    std = float(coerced.std(ddof=1))
    mean = float(coerced.mean())

    if abs(mean) < 1e-10:
        # Mean near zero: use std directly normalized to [0,1] via tanh
        cv = float(np.tanh(std))
    else:
        cv = abs(std / mean)

    if cv < _CONSTANT_CV_THRESHOLD:
        return 0.0, f"variation=0.00 (constant column, CV={cv:.2e})"

    score = round(1.0 - math.exp(-cv), 4)
    return score, f"variation={score:.2f} (CV={cv:.3f})"


def _cardinality_fitness(unique_count: int) -> tuple[float, str]:
    """Return (cardinality fitness score in [0,1], signal string).

    Score is 1.0 within the ideal range, falls off linearly outside it,
    and is 0.0 at or above _MAX_CARDINALITY.

    Documented fitness function:
        k < IDEAL_LOW:      score = k / IDEAL_LOW
        IDEAL_LOW <= k <= IDEAL_HIGH:  score = 1.0
        IDEAL_HIGH < k < MAX:  score = 1 - (k - IDEAL_HIGH) / (MAX - IDEAL_HIGH)
        k >= MAX:           score = 0.0
    """
    k = unique_count
    lo, hi, mx = _IDEAL_CARDINALITY_LOW, _IDEAL_CARDINALITY_HIGH, _MAX_CARDINALITY
    if k >= mx:
        return 0.0, f"cardinality=0.00 (k={k} >= max {mx})"
    if k >= lo and k <= hi:
        return 1.0, f"cardinality=1.00 (k={k} in ideal range [{lo},{hi}])"
    if k < lo:
        score = round(k / lo, 4)
        return score, f"cardinality={score:.2f} (k={k} below ideal low {lo})"
    # k is in (hi, mx)
    score = round(1.0 - (k - hi) / (mx - hi), 4)
    return score, f"cardinality={score:.2f} (k={k} above ideal high {hi})"


def _repetition_score(series: pd.Series, unique_count: int) -> tuple[float, str]:
    """Return (repetition score in [0,1], signal string).

    Repetition score = 1 - unique_ratio  (fraction of non-unique observations).
    A column where every value appears exactly once → score = 0.
    A column where all values are the same → score ≈ 1 (but constant columns
    are handled separately in variation).
    """
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return 0.0, "repetition=0.00 (no non-missing values)"
    unique_ratio = round(unique_count / n, 4)
    rep = round(1.0 - unique_ratio, 4)
    return rep, f"repetition={rep:.2f} ({n - unique_count} repeated observations)"


def _temporal_period_score(series: pd.Series) -> tuple[float, str]:
    """Return (temporal period score in [0,1], signal string).

    Score measures how many distinct usable periods exist relative to a
    threshold of _MIN_TEMPORAL_PERIODS.  Uses saturation:
        score = min(distinct_periods / _MIN_TEMPORAL_PERIODS, 1.0) ... but
    we also normalize by the full range (distinct_count / total_non_null).

    Formula:
        score = min(distinct_count / MIN_TEMPORAL_PERIODS, 1.0)
                * (1 - missing_ratio)
    """
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return 0.0, "temporal_periods=0.00 (no non-missing values)"

    total = len(series)
    missing_ratio = round((total - n) / total, 4)
    distinct = int(series.nunique(dropna=True))

    coverage_factor = round(1.0 - missing_ratio, 4)
    period_factor = round(min(distinct / _MIN_TEMPORAL_PERIODS, 1.0), 4)
    score = round(coverage_factor * period_factor, 4)

    return score, (
        f"temporal_periods={score:.2f} "
        f"({distinct} distinct periods, {coverage_factor:.0%} non-missing)"
    )


# ---------------------------------------------------------------------------
# Per-chart-type scoring functions
# ---------------------------------------------------------------------------

def _score_histogram(
    df: pd.DataFrame,
    candidate: VisualizationCandidate,
) -> tuple[float, list[str]]:
    """Score a HISTOGRAM candidate."""
    x = df[candidate.x_column]
    cov, sig_cov = _coverage(x)
    var, sig_var = _numeric_variation(x)
    w = _W_HISTOGRAM
    score = w["coverage_x"] * cov + w["variation_x"] * var
    signals = [
        sig_cov,
        sig_var,
        f"score_components: coverage×{w['coverage_x']} + variation×{w['variation_x']}",
    ]
    return round(score, 6), signals


def _score_count_bar(
    df: pd.DataFrame,
    candidate: VisualizationCandidate,
    classification: DatasetClassification,
) -> tuple[float, list[str]]:
    """Score a COUNT_BAR candidate."""
    x_col = classification.get_column(candidate.x_column)
    x = df[candidate.x_column]
    cov, sig_cov = _coverage(x)
    unique_count = x_col.unique_count if x_col is not None else int(x.nunique())
    card, sig_card = _cardinality_fitness(unique_count)
    rep, sig_rep = _repetition_score(x, unique_count)
    w = _W_COUNT_BAR
    score = w["coverage_x"] * cov + w["cardinality_x"] * card + w["repetition_x"] * rep
    signals = [
        sig_cov,
        sig_card,
        sig_rep,
        f"score_components: coverage×{w['coverage_x']} + cardinality×{w['cardinality_x']} + repetition×{w['repetition_x']}",
    ]
    return round(score, 6), signals


def _score_bar(
    df: pd.DataFrame,
    candidate: VisualizationCandidate,
    classification: DatasetClassification,
) -> tuple[float, list[str]]:
    """Score a BAR candidate (CATEGORY/BOOLEAN x_column, MEASURE y_column)."""
    x_col = classification.get_column(candidate.x_column)
    x = df[candidate.x_column]
    y = df[candidate.y_column]  # type: ignore[index]

    cov_x, sig_cov_x = _coverage(x)
    unique_count = x_col.unique_count if x_col is not None else int(x.nunique())
    card, sig_card = _cardinality_fitness(unique_count)
    rep, sig_rep = _repetition_score(x, unique_count)
    cov_y, sig_cov_y = _coverage(y)
    var_y, sig_var_y = _numeric_variation(y)

    w = _W_BAR
    score = (
        w["coverage_x"]    * cov_x
        + w["cardinality_x"] * card
        + w["repetition_x"]  * rep
        + w["coverage_y"]    * cov_y
        + w["variation_y"]   * var_y
    )
    signals = [
        sig_cov_x,
        sig_card,
        sig_rep,
        sig_cov_y,
        sig_var_y,
        (
            f"score_components: cov_x×{w['coverage_x']} + card×{w['cardinality_x']} "
            f"+ rep×{w['repetition_x']} + cov_y×{w['coverage_y']} + var_y×{w['variation_y']}"
        ),
    ]
    return round(score, 6), signals


def _score_line(
    df: pd.DataFrame,
    candidate: VisualizationCandidate,
    classification: DatasetClassification,
) -> tuple[float, list[str]]:
    """Score a LINE candidate (TEMPORAL x_column, MEASURE y_column)."""
    x = df[candidate.x_column]
    y = df[candidate.y_column]  # type: ignore[index]

    cov_x, sig_cov_x = _coverage(x)
    temp, sig_temp = _temporal_period_score(x)
    cov_y, sig_cov_y = _coverage(y)
    var_y, sig_var_y = _numeric_variation(y)

    w = _W_LINE
    score = (
        w["coverage_x"]  * cov_x
        + w["temporal_x"]  * temp
        + w["coverage_y"]  * cov_y
        + w["variation_y"] * var_y
    )
    signals = [
        sig_cov_x,
        sig_temp,
        sig_cov_y,
        sig_var_y,
        (
            f"score_components: cov_x×{w['coverage_x']} + temporal×{w['temporal_x']} "
            f"+ cov_y×{w['coverage_y']} + var_y×{w['variation_y']}"
        ),
    ]
    return round(score, 6), signals


def _score_scatter(
    df: pd.DataFrame,
    candidate: VisualizationCandidate,
) -> tuple[float, list[str]]:
    """Score a SCATTER candidate (MEASURE x_column, MEASURE y_column)."""
    x = df[candidate.x_column]
    y = df[candidate.y_column]  # type: ignore[index]

    cov_x, sig_cov_x = _coverage(x)
    var_x, sig_var_x = _numeric_variation(x)
    cov_y, sig_cov_y = _coverage(y)
    var_y, sig_var_y = _numeric_variation(y)

    w = _W_SCATTER
    score = (
        w["coverage_x"] * cov_x
        + w["variation_x"] * var_x
        + w["coverage_y"] * cov_y
        + w["variation_y"] * var_y
    )
    signals = [
        sig_cov_x,
        sig_var_x,
        sig_cov_y,
        sig_var_y,
        (
            f"score_components: cov_x×{w['coverage_x']} + var_x×{w['variation_x']} "
            f"+ cov_y×{w['coverage_y']} + var_y×{w['variation_y']}"
        ),
    ]
    return round(score, 6), signals


# ---------------------------------------------------------------------------
# Sort key for deterministic ordering
# ---------------------------------------------------------------------------

def _sort_key(candidate: RankedCandidate) -> tuple[float, int, str, str]:
    """Sort key: (descending score, chart_type priority, x_column, y_column).

    We negate the score so that sorting ascending gives highest score first.
    y_column None sorts before any string by mapping it to "".
    """
    return (
        -round(candidate.score, 6),
        CHART_TYPE_PRIORITY.get(candidate.chart_type, 99),
        candidate.x_column,
        candidate.y_column if candidate.y_column is not None else "",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_visualization_candidates(
    dataframe: pd.DataFrame,
    classification: DatasetClassification,
    feasibility_result: FeasibilityResult,
) -> RankingResult:
    """Rank Stage 2 visualization candidates by deterministic usefulness score.

    Parameters
    ----------
    dataframe:
        The cleaned/prepared pandas DataFrame.
    classification:
        The Stage 1 ``DatasetClassification`` for the same dataframe.
    feasibility_result:
        The Stage 2 ``FeasibilityResult`` for the same (dataframe, classification).

    Returns
    -------
    RankingResult
        Fully deterministic for a given input triple. Contains ``RankedCandidate``
        items sorted by descending score with deterministic tiebreakers.
    """
    notes: list[str] = []

    # Guard: zero-row dataset
    if len(dataframe) == 0:
        notes.append("Dataset has zero rows — no candidates to rank")
        return RankingResult(
            dataset_name=classification.dataset_name,
            row_count=0,
            column_count=classification.column_count,
            ranked=[],
            ranking_notes=notes,
        )

    # Guard: no feasible candidates from Stage 2
    if not feasibility_result.candidates:
        notes.append("Stage 2 produced no feasible candidates — ranking result is empty")
        return RankingResult(
            dataset_name=classification.dataset_name,
            row_count=len(dataframe),
            column_count=classification.column_count,
            ranked=[],
            ranking_notes=notes,
        )

    # Deduplicate candidates by (chart_type, x_column, y_column) preserving first occurrence.
    seen: set[tuple[str, str, str | None]] = set()
    unique_candidates: list[VisualizationCandidate] = []
    for cand in feasibility_result.candidates:
        key = (cand.chart_type, cand.x_column, cand.y_column)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)
        else:
            notes.append(
                f"Duplicate suppressed: ({cand.chart_type}, {cand.x_column!r}, {cand.y_column!r})"
            )

    # Score each candidate
    pre_ranked: list[RankedCandidate] = []
    for cand in unique_candidates:
        feasibility_signals = list(cand.signals)

        try:
            ct = cand.chart_type
            if ct == "HISTOGRAM":
                score, rank_signals = _score_histogram(dataframe, cand)
            elif ct == "COUNT_BAR":
                score, rank_signals = _score_count_bar(dataframe, cand, classification)
            elif ct == "BAR":
                score, rank_signals = _score_bar(dataframe, cand, classification)
            elif ct == "LINE":
                score, rank_signals = _score_line(dataframe, cand, classification)
            elif ct == "SCATTER":
                score, rank_signals = _score_scatter(dataframe, cand)
            else:
                score = 0.0
                rank_signals = [f"unknown chart type {ct!r} — score defaulted to 0"]
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
            score = 0.0
            rank_signals = [f"scoring error ({type(exc).__name__}: {exc}) — score defaulted to 0"]
            notes.append(
                f"Scoring error for ({cand.chart_type}, {cand.x_column!r}, {cand.y_column!r}): {exc}"
            )

        # Clamp to [0, 1] to guard against floating-point drift
        score = max(0.0, min(1.0, score))

        pre_ranked.append(
            RankedCandidate(
                chart_type=cand.chart_type,
                x_column=cand.x_column,
                y_column=cand.y_column,
                score=round(score, 6),
                rank=0,  # assigned after sorting
                signals=feasibility_signals + rank_signals,
            )
        )

    # Sort deterministically
    pre_ranked.sort(key=_sort_key)

    # Assign 1-based ranks
    ranked: list[RankedCandidate] = [
        RankedCandidate(
            chart_type=c.chart_type,
            x_column=c.x_column,
            y_column=c.y_column,
            score=c.score,
            rank=i + 1,
            signals=c.signals,
        )
        for i, c in enumerate(pre_ranked)
    ]

    return RankingResult(
        dataset_name=classification.dataset_name,
        row_count=len(dataframe),
        column_count=classification.column_count,
        ranked=ranked,
        ranking_notes=notes,
    )

