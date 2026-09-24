"""Deterministic Visualization Feasibility Engine for Pipeline 3 — Results.

Stage 2 of Pipeline 3: given a cleaned pandas DataFrame and the Stage 1
``DatasetClassification``, determine which visualization structures are
*structurally feasible* for the dataset.

This module answers:
    "What graphs can this dataset meaningfully support?"

It does NOT render graphs, produce ranked suggestions, invoke any ML model,
or connect to the API layer.

Design principles
-----------------
* Consumes Stage 1 column roles exclusively — never re-classifies columns.
* Every decision is fully deterministic and explained by explicit signal strings.
* Repeated calls with identical (dataframe, classification) input produce
  identical output.
* Avoids combinatorial explosion: does not blindly cross all possible column
  pairs.  Instead it applies structural rules and capped fan-out limits that
  are documented inline.
* Roles that are structurally forbidden as chart dimensions (IDENTIFIER,
  DESCRIPTIVE_TEXT, UNKNOWN) are never used as axes.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from pipelines.pipeline3_results.visualization.schemas import (
    ColumnClassification,
    DatasetClassification,
    FeasibilityResult,
    VisualizationCandidate,
)

# ---------------------------------------------------------------------------
# Configurable thresholds (all documented)
# ---------------------------------------------------------------------------

# Maximum unique values a CATEGORY (or BOOLEAN treated as category) may have
# before it is considered too high-cardinality to serve as a chart dimension.
# Rationale: bar charts with > this many bars are unreadable.
_MAX_CATEGORY_CARDINALITY: int = 50

# When multiple CATEGORY columns pass the cardinality filter we cap how many
# are paired with each MEASURE to avoid combinatorial explosion.
# e.g. 10 categories × 15 measures = 150 — this cap limits pairings per role.
# Rationale: keeps result set manageable; ranking/selection belongs to Stage 3.
_MAX_CAT_PER_MEASURE: int = 3   # at most 3 categories chosen to pair with each measure
_MAX_MEAS_PER_CAT: int = 3      # at most 3 measures chosen to pair with each category
_MAX_SCATTER_PAIRS: int = 6     # total scatter pairs across the dataset
_MAX_TEMPORAL_MEASURES: int = 3 # measures paired with each temporal column

# A MEASURE column must have at least this many non-null, non-NaN numeric
# observations to participate in a chart.
_MIN_USABLE_NUMERIC_ROWS: int = 2

# A TEMPORAL column must have at least this many non-null observations.
_MIN_USABLE_TEMPORAL_ROWS: int = 2

# A CATEGORY / BOOLEAN column must have at least this many non-null
# observations to participate as a dimension.
_MIN_USABLE_CATEGORY_ROWS: int = 1


# ---------------------------------------------------------------------------
# Internal data-quality inspectors
# ---------------------------------------------------------------------------

def _usable_numeric_count(series: pd.Series) -> int:
    """Return count of non-null, non-infinite numeric observations."""
    non_null = series.dropna()
    if not pd.api.types.is_numeric_dtype(non_null.dtype):
        # Try coercing object / string series
        coerced = pd.to_numeric(non_null, errors="coerce").dropna()
        return len(coerced)
    finite = non_null[~non_null.isin([float("inf"), float("-inf")])]
    return len(finite)


def _usable_temporal_count(series: pd.Series) -> int:
    """Return count of non-null temporal observations."""
    non_null = series.dropna()
    if pd.api.types.is_datetime64_any_dtype(non_null.dtype):
        return len(non_null)
    # String series that might contain dates — already classified TEMPORAL by Stage 1
    coerced = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return int(coerced.notna().sum())


def _usable_category_count(series: pd.Series) -> int:
    """Return count of non-null, non-empty category observations."""
    non_null = series.dropna()
    str_non_empty = non_null.astype(str).str.strip()
    return int((str_non_empty != "").sum())


def _is_measure_usable(df: pd.DataFrame, col: ColumnClassification) -> tuple[bool, str]:
    """Check whether a MEASURE column has sufficient usable rows.

    Returns (usable: bool, reason: str).
    """
    if col.missing_ratio >= 1.0:
        return False, f"{col.name}: all values missing"
    count = _usable_numeric_count(df[col.name])
    if count < _MIN_USABLE_NUMERIC_ROWS:
        return False, f"{col.name}: only {count} usable numeric value(s)"
    return True, ""


def _is_temporal_usable(df: pd.DataFrame, col: ColumnClassification) -> tuple[bool, str]:
    """Check whether a TEMPORAL column has sufficient usable rows."""
    if col.missing_ratio >= 1.0:
        return False, f"{col.name}: all values missing"
    count = _usable_temporal_count(df[col.name])
    if count < _MIN_USABLE_TEMPORAL_ROWS:
        return False, f"{col.name}: only {count} usable temporal value(s)"
    return True, ""


def _is_category_usable(
    df: pd.DataFrame,
    col: ColumnClassification,
    *,
    high_cardinality_limit: int = _MAX_CATEGORY_CARDINALITY,
) -> tuple[bool, str]:
    """Check whether a CATEGORY or BOOLEAN column is usable as a chart dimension."""
    if col.missing_ratio >= 1.0:
        return False, f"{col.name}: all values missing"
    count = _usable_category_count(df[col.name])
    if count < _MIN_USABLE_CATEGORY_ROWS:
        return False, f"{col.name}: no usable category values"
    if col.unique_count > high_cardinality_limit:
        return False, (
            f"{col.name}: cardinality {col.unique_count} exceeds "
            f"dimension limit {high_cardinality_limit}"
        )
    return True, ""


# ---------------------------------------------------------------------------
# Candidate-building helpers
# ---------------------------------------------------------------------------

def _make(
    chart_type: str,
    x_column: str,
    y_column: str | None,
    signals: list[str],
) -> VisualizationCandidate:
    return VisualizationCandidate(
        chart_type=chart_type,  # type: ignore[arg-type]
        x_column=x_column,
        y_column=y_column,
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Rule implementations (one function per structural rule group)
# ---------------------------------------------------------------------------

def _rule_histogram(
    df: pd.DataFrame,
    measures: list[ColumnClassification],
    suppressed: list[str],
) -> list[VisualizationCandidate]:
    """MEASURE alone → HISTOGRAM."""
    candidates: list[VisualizationCandidate] = []
    for m in measures:
        ok, reason = _is_measure_usable(df, m)
        if not ok:
            suppressed.append(f"HISTOGRAM suppressed — {reason}")
            continue
        candidates.append(_make(
            "HISTOGRAM",
            x_column=m.name,
            y_column=None,
            signals=[f"{m.name} is MEASURE", "MEASURE alone → HISTOGRAM"],
        ))
    return candidates


def _rule_count_bar_category(
    df: pd.DataFrame,
    categories: list[ColumnClassification],
    suppressed: list[str],
) -> list[VisualizationCandidate]:
    """CATEGORY alone → COUNT_BAR."""
    candidates: list[VisualizationCandidate] = []
    for cat in categories:
        ok, reason = _is_category_usable(df, cat)
        if not ok:
            suppressed.append(f"COUNT_BAR(cat) suppressed — {reason}")
            continue
        candidates.append(_make(
            "COUNT_BAR",
            x_column=cat.name,
            y_column=None,
            signals=[f"{cat.name} is CATEGORY", "CATEGORY alone → COUNT_BAR"],
        ))
    return candidates


def _rule_count_bar_boolean(
    df: pd.DataFrame,
    booleans: list[ColumnClassification],
    suppressed: list[str],
) -> list[VisualizationCandidate]:
    """BOOLEAN alone → COUNT_BAR (treated as low-cardinality category)."""
    candidates: list[VisualizationCandidate] = []
    for b in booleans:
        ok, reason = _is_category_usable(df, b, high_cardinality_limit=2)
        if not ok:
            suppressed.append(f"COUNT_BAR(bool) suppressed — {reason}")
            continue
        candidates.append(_make(
            "COUNT_BAR",
            x_column=b.name,
            y_column=None,
            signals=[
                f"{b.name} is BOOLEAN",
                "BOOLEAN treated as low-cardinality CATEGORY → COUNT_BAR",
            ],
        ))
    return candidates


def _rule_bar_category_measure(
    df: pd.DataFrame,
    categories: list[ColumnClassification],
    measures: list[ColumnClassification],
    suppressed: list[str],
) -> list[VisualizationCandidate]:
    """CATEGORY + MEASURE → BAR.

    Anti-explosion rule: for each category, take at most _MAX_MEAS_PER_CAT
    measures; for each measure, take at most _MAX_CAT_PER_MEASURE categories.
    Combined, this bounds total pairs to a manageable set.
    """
    candidates: list[VisualizationCandidate] = []

    # Pre-filter to usable columns
    usable_cats: list[ColumnClassification] = []
    for cat in categories:
        ok, reason = _is_category_usable(df, cat)
        if ok:
            usable_cats.append(cat)
        else:
            suppressed.append(f"BAR(cat×meas) category suppressed — {reason}")

    usable_meas: list[ColumnClassification] = []
    for m in measures:
        ok, reason = _is_measure_usable(df, m)
        if ok:
            usable_meas.append(m)
        else:
            suppressed.append(f"BAR(cat×meas) measure suppressed — {reason}")

    if not usable_cats or not usable_meas:
        return candidates

    # Cap fan-out: iterate over measures, pick up to _MAX_CAT_PER_MEASURE cats each
    meas_pair_count: dict[str, int] = {}
    cat_pair_count: dict[str, int] = {}

    for m in usable_meas:
        for cat in usable_cats:
            if meas_pair_count.get(m.name, 0) >= _MAX_CAT_PER_MEASURE:
                break
            if cat_pair_count.get(cat.name, 0) >= _MAX_MEAS_PER_CAT:
                continue
            candidates.append(_make(
                "BAR",
                x_column=cat.name,
                y_column=m.name,
                signals=[
                    f"{cat.name} is CATEGORY",
                    f"{m.name} is MEASURE",
                    "CATEGORY + MEASURE → BAR",
                ],
            ))
            meas_pair_count[m.name] = meas_pair_count.get(m.name, 0) + 1
            cat_pair_count[cat.name] = cat_pair_count.get(cat.name, 0) + 1

    return candidates


def _rule_bar_boolean_measure(
    df: pd.DataFrame,
    booleans: list[ColumnClassification],
    measures: list[ColumnClassification],
    suppressed: list[str],
) -> list[VisualizationCandidate]:
    """BOOLEAN + MEASURE → BAR (BOOLEAN treated as low-cardinality category)."""
    candidates: list[VisualizationCandidate] = []

    usable_bools: list[ColumnClassification] = []
    for b in booleans:
        ok, reason = _is_category_usable(df, b, high_cardinality_limit=2)
        if ok:
            usable_bools.append(b)
        else:
            suppressed.append(f"BAR(bool×meas) boolean suppressed — {reason}")

    usable_meas: list[ColumnClassification] = []
    for m in measures:
        ok, reason = _is_measure_usable(df, m)
        if ok:
            usable_meas.append(m)
        else:
            # Already reported in histogram/bar-cat rules; avoid duplicating suppression
            pass

    if not usable_bools or not usable_meas:
        return candidates

    meas_pair_count: dict[str, int] = {}
    for b in usable_bools:
        for m in usable_meas:
            if meas_pair_count.get(m.name, 0) >= _MAX_CAT_PER_MEASURE:
                break
            candidates.append(_make(
                "BAR",
                x_column=b.name,
                y_column=m.name,
                signals=[
                    f"{b.name} is BOOLEAN",
                    f"{m.name} is MEASURE",
                    "BOOLEAN + MEASURE → BAR",
                ],
            ))
            meas_pair_count[m.name] = meas_pair_count.get(m.name, 0) + 1

    return candidates


def _rule_line_temporal_measure(
    df: pd.DataFrame,
    temporals: list[ColumnClassification],
    measures: list[ColumnClassification],
    suppressed: list[str],
) -> list[VisualizationCandidate]:
    """TEMPORAL + MEASURE → LINE."""
    candidates: list[VisualizationCandidate] = []

    usable_temps: list[ColumnClassification] = []
    for t in temporals:
        ok, reason = _is_temporal_usable(df, t)
        if ok:
            usable_temps.append(t)
        else:
            suppressed.append(f"LINE suppressed — temporal: {reason}")

    usable_meas: list[ColumnClassification] = []
    for m in measures:
        ok, reason = _is_measure_usable(df, m)
        if ok:
            usable_meas.append(m)
        # suppression already noted in histogram/bar rules

    if not usable_temps or not usable_meas:
        return candidates

    for t in usable_temps:
        meas_count = 0
        for m in usable_meas:
            if meas_count >= _MAX_TEMPORAL_MEASURES:
                break
            candidates.append(_make(
                "LINE",
                x_column=t.name,
                y_column=m.name,
                signals=[
                    f"{t.name} is TEMPORAL",
                    f"{m.name} is MEASURE",
                    "TEMPORAL + MEASURE → LINE",
                ],
            ))
            meas_count += 1

    return candidates


def _rule_scatter_measure_measure(
    df: pd.DataFrame,
    measures: list[ColumnClassification],
    suppressed: list[str],
) -> list[VisualizationCandidate]:
    """MEASURE + MEASURE → SCATTER.

    Anti-explosion: only generate up to _MAX_SCATTER_PAIRS unique (a, b) pairs,
    respecting that (a, b) and (b, a) are the same scatter.
    """
    candidates: list[VisualizationCandidate] = []

    usable: list[ColumnClassification] = []
    for m in measures:
        ok, _ = _is_measure_usable(df, m)
        if ok:
            usable.append(m)

    if len(usable) < 2:
        return candidates

    pair_count = 0
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            if pair_count >= _MAX_SCATTER_PAIRS:
                suppressed.append(
                    f"SCATTER: additional pairs suppressed (limit {_MAX_SCATTER_PAIRS})"
                )
                return candidates
            candidates.append(_make(
                "SCATTER",
                x_column=a.name,
                y_column=b.name,
                signals=[
                    f"{a.name} is MEASURE",
                    f"{b.name} is MEASURE",
                    "MEASURE + MEASURE → SCATTER",
                ],
            ))
            pair_count += 1

    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_visualization_feasibility(
    dataframe: pd.DataFrame,
    classification: DatasetClassification,
) -> FeasibilityResult:
    """Determine which visualization structures are feasible for a dataset.

    Consumes an existing Stage 1 ``DatasetClassification`` — never re-classifies
    columns. Returns a ``FeasibilityResult`` containing all feasible
    ``VisualizationCandidate`` entries and a log of suppressed combinations.

    Parameters
    ----------
    dataframe:
        The cleaned/prepared pandas DataFrame corresponding to the
        classification.
    classification:
        The Stage 1 ``DatasetClassification`` for the same dataframe.

    Returns
    -------
    FeasibilityResult
        Fully deterministic for a given (dataframe, classification) pair.
    """
    suppressed: list[str] = []

    # -----------------------------------------------------------------------
    # Guard: zero-row dataset
    # -----------------------------------------------------------------------
    if len(dataframe) == 0:
        suppressed.append("Dataset has zero rows — no visualizations feasible")
        return FeasibilityResult(
            dataset_name=classification.dataset_name,
            row_count=0,
            column_count=classification.column_count,
            candidates=[],
            suppressed_reasons=suppressed,
        )

    # -----------------------------------------------------------------------
    # Partition columns by role (consume Stage 1 results)
    # -----------------------------------------------------------------------
    measures = classification.columns_by_role("MEASURE")
    categories = classification.columns_by_role("CATEGORY")
    temporals = classification.columns_by_role("TEMPORAL")
    booleans = classification.columns_by_role("BOOLEAN")

    # IDENTIFIER, DESCRIPTIVE_TEXT, UNKNOWN are intentionally excluded from
    # all chart combinations.  Log suppression once per role group.
    _ids = classification.columns_by_role("IDENTIFIER")
    _texts = classification.columns_by_role("DESCRIPTIVE_TEXT")
    _unknowns = classification.columns_by_role("UNKNOWN")
    for col in _ids:
        suppressed.append(
            f"IDENTIFIER {col.name!r} excluded — identifiers must not be chart dimensions"
        )
    for col in _texts:
        suppressed.append(
            f"DESCRIPTIVE_TEXT {col.name!r} excluded — free-form text must not be chart dimensions"
        )
    for col in _unknowns:
        suppressed.append(
            f"UNKNOWN {col.name!r} excluded — classification unknown, cannot determine feasibility"
        )

    # -----------------------------------------------------------------------
    # Apply structural rules (order determines final candidate ordering)
    # -----------------------------------------------------------------------
    candidates: list[VisualizationCandidate] = []

    # Rule 1: MEASURE alone → HISTOGRAM
    candidates.extend(_rule_histogram(dataframe, measures, suppressed))

    # Rule 2: CATEGORY alone → COUNT_BAR
    candidates.extend(_rule_count_bar_category(dataframe, categories, suppressed))

    # Rule 3: BOOLEAN alone → COUNT_BAR
    candidates.extend(_rule_count_bar_boolean(dataframe, booleans, suppressed))

    # Rule 4: CATEGORY + MEASURE → BAR
    candidates.extend(_rule_bar_category_measure(dataframe, categories, measures, suppressed))

    # Rule 5: BOOLEAN + MEASURE → BAR
    candidates.extend(_rule_bar_boolean_measure(dataframe, booleans, measures, suppressed))

    # Rule 6: TEMPORAL + MEASURE → LINE
    candidates.extend(_rule_line_temporal_measure(dataframe, temporals, measures, suppressed))

    # Rule 7: MEASURE + MEASURE → SCATTER
    candidates.extend(_rule_scatter_measure_measure(dataframe, measures, suppressed))

    return FeasibilityResult(
        dataset_name=classification.dataset_name,
        row_count=len(dataframe),
        column_count=classification.column_count,
        candidates=candidates,
        suppressed_reasons=suppressed,
    )

