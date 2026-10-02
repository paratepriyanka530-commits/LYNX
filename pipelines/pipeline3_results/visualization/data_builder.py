"""Deterministic Chart Data Generation — Pipeline 3, Stage 4.

Stage 4 of Pipeline 3: given a cleaned pandas DataFrame, the Stage 1
``DatasetClassification``, and a single Stage 3 ``RankedCandidate``,
produce the structured, JSON-serializable chart data payload ready for
any chart renderer.

This module answers:
    "Given this ranked visualization candidate, what exact data should
    the chart renderer plot?"

It does NOT:
  - re-classify columns (Stage 1's job)
  - re-discover feasibility (Stage 2's job)
  - re-rank candidates (Stage 3's job)
  - render charts or create Plotly / Matplotlib figures
  - call any ML model, NLP, LLM, RAG, embeddings, or external service

Design principles
-----------------
* Consumes a Stage 3 ``RankedCandidate`` exclusively — never generates
  its own candidates.
* All output values are native Python scalars (int, float, str, None) so
  the result is directly JSON-serializable.
* Every operation is deterministic: identical inputs → identical outputs.
* Missing, non-finite, and invalid values are filtered out, never fabricated.
* Hard size limits (documented below) prevent enormous payloads; when
  limiting is necessary a deterministic stride-based strategy is used —
  never random sampling.

Aggregation selection for BAR / LINE
-------------------------------------
The appropriate measure aggregation is selected using documented rules:

  1. Pipeline 1 semantic evidence (``numeric_meaning`` / ``value_function``)
     is checked first:
       - ``"count"`` / ``"discrete_count"`` / ``"cumulative"``      → SUM
       - ``"currency"`` / ``"continuous_quantity"``                  → SUM
       - ``"percentage"`` / ``"ratio"`` / ``"rate"`` / ``"score"``  → MEAN
  2. Column name token matching (additive terms → SUM; rate terms → MEAN).
  3. Conservative default: MEAN (never silently assume additivity).

Size limits (all documented here and in schemas.py)
-----------------------------------------------------
* ``STAGE4_MAX_CATEGORY_POINTS = 50``  — BAR / COUNT_BAR: max bars.
  Rationale: more than 50 categories produce unreadable bar charts.
  Limiting strategy: keep the top-N categories by observation count.

* ``STAGE4_MAX_LINE_POINTS = 500``     — LINE: max chronological points.
  Rationale: 500 points is sufficient for readable time-series charts;
  beyond this the lines overlap visually.
  Limiting strategy: deterministic stride (evenly-spaced by index).

* ``STAGE4_MAX_SCATTER_POINTS = 2000`` — SCATTER: max row pairs.
  Rationale: scatter plots with > 2000 points become illegible overplotting.
  Limiting strategy: deterministic stride.

* ``STAGE4_MAX_HISTOGRAM_BINS = 50``   — HISTOGRAM: max bins.
  Rationale: 50 bins is the upper bound for readable histograms; Sturges'
  rule (see below) typically produces far fewer bins.

Histogram binning rule (Sturges' rule, capped)
-----------------------------------------------
    n_bins = min(ceil(log2(N) + 1), STAGE4_MAX_HISTOGRAM_BINS)
  where N = number of usable (finite, non-missing) observations.
  Empty bins outside the observed data range are excluded.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from pipelines.pipeline3_results.visualization.schemas import (
    AggregationMethod,
    ChartData,
    ChartPoint,
    DatasetClassification,
    HistogramBin,
    RankedCandidate,
    STAGE4_MAX_CATEGORY_POINTS,
    STAGE4_MAX_HISTOGRAM_BINS,
    STAGE4_MAX_LINE_POINTS,
    STAGE4_MAX_SCATTER_POINTS,
    VisualizationType,
)

# ---------------------------------------------------------------------------
# Name-token vocabularies for aggregation selection
# ---------------------------------------------------------------------------

_SUM_NAME_TOKENS: frozenset[str] = frozenset({
    "total", "sum", "amount", "revenue", "sales", "count", "qty",
    "quantity", "volume", "units", "sold", "income", "profit", "spend",
    "expenses", "cost", "fee", "tax", "payment", "budget", "balance",
    "gross", "net", "cumulative", "purchases",
})

_MEAN_NAME_TOKENS: frozenset[str] = frozenset({
    "rate", "ratio", "pct", "percent", "percentage", "avg", "average",
    "mean", "score", "rating", "index", "factor", "coefficient",
    "margin", "yield", "density", "frequency", "proportion", "efficiency",
})

_SUM_SEMANTIC_SIGNALS: frozenset[str] = frozenset({
    "count", "discrete_count", "cumulative", "currency", "continuous_quantity",
})

_MEAN_SEMANTIC_SIGNALS: frozenset[str] = frozenset({
    "percentage", "ratio", "rate", "score", "proportion",
})


# ---------------------------------------------------------------------------
# Aggregation selection
# ---------------------------------------------------------------------------

def _name_tokens(col_name: str) -> list[str]:
    """Tokenize a column name into lowercase words."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col_name)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.casefold()).strip("_")
    return [t for t in s.split("_") if t]


def _select_aggregation(
    col_name: str,
    classification: DatasetClassification,
) -> tuple[AggregationMethod, str]:
    """Choose SUM or MEAN for the given measure column.

    Returns (aggregation, reason_signal).
    """
    col_info = classification.get_column(col_name)
    if col_info is not None:
        # Check Pipeline 1 semantic evidence embedded in signals
        # (Stage 1 stores evidence as signal strings containing the raw values)
        # Also look at the column's role signals for numeric_meaning / value_function
        for signal in col_info.signals:
            sig_lower = signal.lower()
            for kw in _SUM_SEMANTIC_SIGNALS:
                if kw in sig_lower:
                    return "sum", f"{col_name} aggregated using sum (Pipeline 1 semantic: {kw})"
            for kw in _MEAN_SEMANTIC_SIGNALS:
                if kw in sig_lower:
                    return "mean", f"{col_name} aggregated using mean (Pipeline 1 semantic: {kw})"

    # Name-token matching
    tokens = set(_name_tokens(col_name))
    if tokens & _SUM_NAME_TOKENS:
        matched = sorted(tokens & _SUM_NAME_TOKENS)
        return "sum", f"{col_name} aggregated using sum (name tokens: {matched})"
    if tokens & _MEAN_NAME_TOKENS:
        matched = sorted(tokens & _MEAN_NAME_TOKENS)
        return "mean", f"{col_name} aggregated using mean (name tokens: {matched})"

    # Conservative default
    return "mean", f"{col_name} aggregated using mean (conservative default — no additive evidence)"


# ---------------------------------------------------------------------------
# Guard helpers
# ---------------------------------------------------------------------------

def _no_data(
    candidate: RankedCandidate,
    reason: str,
    extra_signals: list[str] | None = None,
) -> ChartData:
    """Return a ChartData with has_data=False and a descriptive signal."""
    signals = [reason]
    if extra_signals:
        signals.extend(extra_signals)
    return ChartData(
        chart_type=candidate.chart_type,
        x_column=candidate.x_column,
        y_column=candidate.y_column,
        aggregation=None,
        points=[],
        bins=[],
        signals=signals,
        usable_row_count=0,
        sampled=False,
        has_data=False,
    )


def _to_py_scalar(val: Any) -> float | int | str:
    """Convert numpy / pandas scalar to a native Python scalar."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, float):
        return val
    if isinstance(val, int):
        return val
    return str(val)


def _safe_float(val: Any) -> float | None:
    """Return a finite float or None."""
    try:
        f = float(val)
        if math.isfinite(f):
            return f
    except (TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# BAR builder
# ---------------------------------------------------------------------------

def _build_bar(
    df: pd.DataFrame,
    candidate: RankedCandidate,
    classification: DatasetClassification,
) -> ChartData:
    x_col = candidate.x_column
    y_col = candidate.y_column  # type: ignore[assignment]
    signals: list[str] = []

    # Validate columns exist
    if x_col not in df.columns:
        return _no_data(candidate, f"x_column {x_col!r} not found in dataframe")
    if y_col not in df.columns:
        return _no_data(candidate, f"y_column {y_col!r} not found in dataframe")

    # Drop rows missing either column
    valid = df[[x_col, y_col]].copy()
    valid[y_col] = pd.to_numeric(valid[y_col], errors="coerce")
    initial_rows = len(valid)
    valid = valid.dropna(subset=[x_col, y_col])

    # Guard: if all rows were dropped, return early before chained filters fail
    if len(valid) == 0:
        signals.append(f"{initial_rows} rows excluded (all missing or non-numeric y values)")
        signals.append("0 valid observations used")
        return _no_data(candidate,
                        f"no usable rows after filtering for BAR({x_col}, {y_col})", signals)

    # Remove non-finite y values
    valid = valid[valid[y_col].apply(lambda v: math.isfinite(float(v)))]
    # Remove empty-string x values
    valid = valid[valid[x_col].astype(str).str.strip() != ""]
    usable = len(valid)
    dropped = initial_rows - usable
    if dropped:
        signals.append(f"{dropped} rows excluded (missing or non-finite values)")
    signals.append(f"{usable} valid observations used")

    if usable == 0:
        return _no_data(candidate, f"no usable rows after filtering for BAR({x_col}, {y_col})",
                        signals)

    # Select aggregation
    agg, agg_signal = _select_aggregation(y_col, classification)
    signals.append(agg_signal)

    # Group and aggregate
    grouped = valid.groupby(x_col, sort=False)[y_col]
    if agg == "sum":
        agg_series = grouped.sum()
    else:
        agg_series = grouped.mean()

    # Determine ordering: by aggregated value descending, then lexical x tie-break
    agg_df = agg_series.reset_index()
    agg_df.columns = ["x", "y"]
    agg_df = agg_df.sort_values(
        by=["y", "x"], ascending=[False, True]
    ).reset_index(drop=True)

    # Apply size limit
    sampled = False
    total_groups = len(agg_df)
    if total_groups > STAGE4_MAX_CATEGORY_POINTS:
        agg_df = agg_df.iloc[:STAGE4_MAX_CATEGORY_POINTS]
        signals.append(
            f"truncated to top {STAGE4_MAX_CATEGORY_POINTS} categories "
            f"(from {total_groups}) by {agg} value"
        )
        sampled = True
    signals.append(f"{len(agg_df)} category groups produced")

    points = [
        ChartPoint(x=_to_py_scalar(row["x"]), y=float(row["y"]))
        for _, row in agg_df.iterrows()
    ]

    return ChartData(
        chart_type="BAR",
        x_column=x_col,
        y_column=y_col,
        aggregation=agg,
        points=points,
        bins=[],
        signals=signals,
        usable_row_count=usable,
        sampled=sampled,
        has_data=True,
    )


# ---------------------------------------------------------------------------
# COUNT_BAR builder
# ---------------------------------------------------------------------------

def _build_count_bar(
    df: pd.DataFrame,
    candidate: RankedCandidate,
) -> ChartData:
    x_col = candidate.x_column
    signals: list[str] = []

    if x_col not in df.columns:
        return _no_data(candidate, f"x_column {x_col!r} not found in dataframe")

    x_series = df[x_col].dropna()
    x_series = x_series[x_series.astype(str).str.strip() != ""]
    usable = len(x_series)

    if usable == 0:
        return _no_data(candidate, f"no usable values in {x_col!r} for COUNT_BAR", signals)

    signals.append(f"{usable} valid category observations used")

    counts = x_series.astype(str).value_counts(sort=True, ascending=False)
    # Tie-break: sort by (count descending, label ascending) for determinism
    counts_df = (
        counts.reset_index()
    )
    counts_df.columns = ["x", "y"]
    counts_df = counts_df.sort_values(
        by=["y", "x"], ascending=[False, True]
    ).reset_index(drop=True)

    sampled = False
    total_groups = len(counts_df)
    if total_groups > STAGE4_MAX_CATEGORY_POINTS:
        counts_df = counts_df.iloc[:STAGE4_MAX_CATEGORY_POINTS]
        signals.append(
            f"truncated to top {STAGE4_MAX_CATEGORY_POINTS} categories "
            f"(from {total_groups}) by count"
        )
        sampled = True
    signals.append(f"{len(counts_df)} category groups produced")

    points = [
        ChartPoint(x=str(row["x"]), y=int(row["y"]))
        for _, row in counts_df.iterrows()
    ]

    return ChartData(
        chart_type="COUNT_BAR",
        x_column=x_col,
        y_column=None,
        aggregation="count",
        points=points,
        bins=[],
        signals=signals,
        usable_row_count=usable,
        sampled=sampled,
        has_data=True,
    )


# ---------------------------------------------------------------------------
# LINE builder
# ---------------------------------------------------------------------------

def _choose_temporal_granularity(
    parsed: pd.Series,
) -> tuple[str, str]:
    """Choose a deterministic pandas frequency alias for resampling.

    Rules (documented):
      - < 2 distinct days              → keep raw (no resampling)
      - date range ≤ 90 days           → daily ('D')
      - date range ≤ 730 days (~2 yr)  → weekly ('W')
      - date range ≤ 1825 days (~5 yr) → monthly ('ME')
      - date range > 1825 days         → quarterly ('QE')

    Returns (pandas_freq_alias, human_label).
    """
    distinct = parsed.nunique()
    if distinct < 2:
        return "", "raw (< 2 distinct periods)"

    span_days = (parsed.max() - parsed.min()).days
    if span_days <= 90:
        return "D", "daily"
    if span_days <= 730:
        return "W", "weekly"
    if span_days <= 1825:
        return "ME", "monthly"
    return "QE", "quarterly"


def _build_line(
    df: pd.DataFrame,
    candidate: RankedCandidate,
    classification: DatasetClassification,
) -> ChartData:
    x_col = candidate.x_column
    y_col = candidate.y_column  # type: ignore[assignment]
    signals: list[str] = []

    if x_col not in df.columns:
        return _no_data(candidate, f"x_column {x_col!r} not found in dataframe")
    if y_col not in df.columns:
        return _no_data(candidate, f"y_column {y_col!r} not found in dataframe")

    working = df[[x_col, y_col]].copy()
    working[y_col] = pd.to_numeric(working[y_col], errors="coerce")

    # Parse temporal column
    x_raw = working[x_col]
    if pd.api.types.is_datetime64_any_dtype(x_raw.dtype):
        parsed_x = x_raw
    else:
        parsed_x = pd.to_datetime(x_raw, errors="coerce", format="mixed")

    initial_rows = len(working)
    valid_mask = parsed_x.notna() & working[y_col].notna() & working[y_col].apply(
        lambda v: math.isfinite(float(v)) if pd.notna(v) else False
    )
    invalid_temporal = int((parsed_x.isna()).sum())
    working["_x_parsed"] = parsed_x
    working = working[valid_mask].copy()

    if invalid_temporal:
        signals.append(f"{invalid_temporal} invalid temporal values excluded")

    dropped_y = initial_rows - len(working) - invalid_temporal
    if dropped_y > 0:
        signals.append(f"{dropped_y} rows excluded (missing/non-finite measure values)")

    usable = len(working)
    signals.append(f"{usable} valid observations used")

    if usable == 0:
        return _no_data(candidate, f"no usable rows for LINE({x_col}, {y_col})", signals)

    # Choose aggregation for the measure
    agg, agg_signal = _select_aggregation(y_col, classification)
    signals.append(agg_signal)

    # Choose granularity
    freq, freq_label = _choose_temporal_granularity(working["_x_parsed"])
    signals.append(f"temporal granularity: {freq_label}")

    if freq:
        # Resample: set index to parsed datetime, resample, aggregate
        ts = working.set_index("_x_parsed")[y_col]
        if agg == "sum":
            resampled = ts.resample(freq).sum()
        else:
            resampled = ts.resample(freq).mean()
        resampled = resampled.dropna()
        agg_df = resampled.reset_index()
        agg_df.columns = ["_x_parsed", y_col]
        signals.append(f"resampled to {freq_label} ({len(agg_df)} periods)")
    else:
        # No resampling: sort by datetime, keep as-is
        working_sorted = working.sort_values("_x_parsed").reset_index(drop=True)
        agg_df = working_sorted[["_x_parsed", y_col]].copy()
        signals.append(f"no resampling applied ({len(agg_df)} raw periods)")

    # Sort chronologically
    agg_df = agg_df.sort_values("_x_parsed").reset_index(drop=True)

    # Apply size limit with deterministic stride
    sampled = False
    total_points = len(agg_df)
    if total_points > STAGE4_MAX_LINE_POINTS:
        stride = math.ceil(total_points / STAGE4_MAX_LINE_POINTS)
        agg_df = agg_df.iloc[::stride].reset_index(drop=True)
        signals.append(
            f"deterministically sampled from {total_points} to {len(agg_df)} points "
            f"(stride={stride})"
        )
        sampled = True

    # Convert datetime to ISO string for JSON serializability
    points = []
    for _, row in agg_df.iterrows():
        x_val = row["_x_parsed"]
        y_val = row[y_col]
        x_str = x_val.isoformat() if hasattr(x_val, "isoformat") else str(x_val)
        y_f = _safe_float(y_val)
        if y_f is None:
            continue
        points.append(ChartPoint(x=x_str, y=y_f))

    return ChartData(
        chart_type="LINE",
        x_column=x_col,
        y_column=y_col,
        aggregation=agg,
        points=points,
        bins=[],
        signals=signals,
        usable_row_count=usable,
        sampled=sampled,
        has_data=len(points) > 0,
    )


# ---------------------------------------------------------------------------
# SCATTER builder
# ---------------------------------------------------------------------------

def _build_scatter(
    df: pd.DataFrame,
    candidate: RankedCandidate,
) -> ChartData:
    x_col = candidate.x_column
    y_col = candidate.y_column  # type: ignore[assignment]
    signals: list[str] = []

    if x_col not in df.columns:
        return _no_data(candidate, f"x_column {x_col!r} not found in dataframe")
    if y_col not in df.columns:
        return _no_data(candidate, f"y_column {y_col!r} not found in dataframe")

    working = df[[x_col, y_col]].copy()
    working[x_col] = pd.to_numeric(working[x_col], errors="coerce")
    working[y_col] = pd.to_numeric(working[y_col], errors="coerce")

    initial_rows = len(working)
    # Keep only rows where both x and y are finite
    mask = working[x_col].notna() & working[y_col].notna()
    mask &= working[x_col].apply(lambda v: math.isfinite(float(v)) if pd.notna(v) else False)
    mask &= working[y_col].apply(lambda v: math.isfinite(float(v)) if pd.notna(v) else False)
    working = working[mask].reset_index(drop=True)
    usable = len(working)
    dropped = initial_rows - usable
    if dropped:
        signals.append(f"{dropped} rows excluded (missing or non-finite values)")
    signals.append(f"{usable} valid observations used")

    if usable == 0:
        return _no_data(candidate, f"no usable rows for SCATTER({x_col}, {y_col})", signals)

    # Deterministic stride-based limiting
    sampled = False
    total_points = usable
    if total_points > STAGE4_MAX_SCATTER_POINTS:
        stride = math.ceil(total_points / STAGE4_MAX_SCATTER_POINTS)
        working = working.iloc[::stride].reset_index(drop=True)
        signals.append(
            f"deterministically sampled from {total_points} to {len(working)} points "
            f"(stride={stride}; sampled points do not represent the full dataset)"
        )
        sampled = True

    points = [
        ChartPoint(x=float(row[x_col]), y=float(row[y_col]))
        for _, row in working.iterrows()
    ]

    return ChartData(
        chart_type="SCATTER",
        x_column=x_col,
        y_column=y_col,
        aggregation=None,
        points=points,
        bins=[],
        signals=signals,
        usable_row_count=usable,
        sampled=sampled,
        has_data=True,
    )


# ---------------------------------------------------------------------------
# HISTOGRAM builder
# ---------------------------------------------------------------------------

def _build_histogram(
    df: pd.DataFrame,
    candidate: RankedCandidate,
) -> ChartData:
    x_col = candidate.x_column
    signals: list[str] = []

    if x_col not in df.columns:
        return _no_data(candidate, f"x_column {x_col!r} not found in dataframe")

    x_numeric = pd.to_numeric(df[x_col], errors="coerce")
    initial_rows = len(x_numeric)
    x_finite = x_numeric[x_numeric.notna() & x_numeric.apply(
        lambda v: math.isfinite(float(v)) if pd.notna(v) else False
    )]
    usable = len(x_finite)
    dropped = initial_rows - usable
    if dropped:
        signals.append(f"{dropped} values excluded (missing or non-finite)")
    signals.append(f"{usable} valid numeric observations used")

    if usable == 0:
        return _no_data(candidate, f"no usable numeric values in {x_col!r} for HISTOGRAM",
                        signals)

    # Sturges' rule: n_bins = ceil(log2(N) + 1), capped at STAGE4_MAX_HISTOGRAM_BINS
    n_bins = min(math.ceil(math.log2(usable) + 1), STAGE4_MAX_HISTOGRAM_BINS)
    signals.append(
        f"histogram generated using {n_bins} bins (Sturges' rule: "
        f"ceil(log2({usable}) + 1) = {math.ceil(math.log2(usable) + 1)}, "
        f"cap = {STAGE4_MAX_HISTOGRAM_BINS})"
    )

    # Compute histogram using numpy (deterministic)
    vals = x_finite.values.astype(float)
    counts_arr, edges = np.histogram(vals, bins=n_bins)

    # Build HistogramBin list, excluding empty bins outside observed range
    bins: list[HistogramBin] = []
    for i, cnt in enumerate(counts_arr):
        if cnt > 0:
            bins.append(HistogramBin(
                bin_start=round(float(edges[i]), 10),
                bin_end=round(float(edges[i + 1]), 10),
                count=int(cnt),
            ))
    signals.append(f"{len(bins)} non-empty bins produced")

    return ChartData(
        chart_type="HISTOGRAM",
        x_column=x_col,
        y_column=None,
        aggregation="count",
        points=[],
        bins=bins,
        signals=signals,
        usable_row_count=usable,
        sampled=False,
        has_data=len(bins) > 0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_chart_data(
    dataframe: pd.DataFrame,
    classification: DatasetClassification,
    candidate: RankedCandidate,
) -> ChartData:
    """Generate deterministic, JSON-serializable chart data for a Stage 3 candidate.

    Parameters
    ----------
    dataframe:
        The cleaned/prepared pandas DataFrame.
    classification:
        The Stage 1 ``DatasetClassification`` for the same dataframe.
    candidate:
        A single Stage 3 ``RankedCandidate``.

    Returns
    -------
    ChartData
        Fully deterministic for a given input triple.  If no usable chart
        data can be produced, returns a ``ChartData`` with ``has_data=False``
        and deterministic signals explaining why.
    """
    # Guard: empty dataframe
    if len(dataframe) == 0:
        return _no_data(candidate, "dataframe has zero rows — no chart data can be produced")

    ct = candidate.chart_type
    if ct == "BAR":
        return _build_bar(dataframe, candidate, classification)
    if ct == "COUNT_BAR":
        return _build_count_bar(dataframe, candidate)
    if ct == "LINE":
        return _build_line(dataframe, candidate, classification)
    if ct == "SCATTER":
        return _build_scatter(dataframe, candidate)
    if ct == "HISTOGRAM":
        return _build_histogram(dataframe, candidate)
    if ct == "PIE":
        base = (
            _build_bar(dataframe, candidate, classification)
            if candidate.y_column is not None
            else _build_count_bar(dataframe, candidate)
        )
        return base.model_copy(update={
            "chart_type": "PIE",
            "signals": [*base.signals, "PIE part-to-whole chart data prepared"],
        })

    return _no_data(candidate, f"unsupported chart type: {ct!r}")
