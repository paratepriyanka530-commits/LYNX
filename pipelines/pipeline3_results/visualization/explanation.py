"""Deterministic Visualization Explanation & XAI Evidence — Pipeline 3, Stage 9.

Stage 9 of Pipeline 3: generates factual, auditable, deterministic explanations
for already-generated visualization payloads (Stage 6 automatic payloads or
Stage 8 requested visualization payloads).

Design Principles:
* Strictly Factual: Uses ONLY information already present in the visualization payload / chart_data.
* Zero Statistical Overreach: Never infers causality, correlation, importance, popularity,
  preference, prediction, or statistical significance.
* Attribution Discipline: Clearly distinguishes between user-requested aggregations (Stage 8)
  and automatic semantic/default aggregations (Stage 5/6).
* Non-Crashing Resilience: Gracefully handles missing chart data, empty payloads,
  INVALID payloads, and NEEDS_CLARIFICATION payloads without exceptions.
* Deterministic Output: Identical inputs produce identical outputs with zero randomness.
* Pure JSON Safety: Returns standard Python primitives (dict, list, str, int, float, bool, None).
* Non-Mutating: Never alters the input payload.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from pipelines.pipeline3_results.visualization.integration import _to_json_safe

# Forbidden terms: must never appear in generated explanations
FORBIDDEN_WORDS: frozenset[str] = frozenset({
    "causes",
    "cause",
    "causing",
    "caused",
    "because",
    "drives",
    "driving",
    "proves",
    "proving",
    "significant",
    "significance",
    "important",
    "importance",
    "best",
    "worst",
    "popular",
    "popularity",
    "preferred",
    "preference",
    "predicts",
    "predicting",
    "prediction",
    "correlated",
    "correlates",
    "correlation",
})


def _fmt_num(val: Any) -> str:
    """Format numeric value with commas, preserving integers cleanly."""
    if val is None:
        return "None"
    if isinstance(val, (int, np.integer)):
        return f"{int(val):,}"
    if isinstance(val, (float, np.floating)):
        fval = float(val)
        if fval.is_integer():
            return f"{int(fval):,}"
        # Format up to 2 decimal places, removing trailing zeros
        return f"{fval:,.2f}".rstrip("0").rstrip(".")
    return str(val)


def _determine_aggregation_attribution(
    raw_agg: str | None,
    signals: list[str],
    has_score: bool,
) -> str:
    """Determine the factual attribution statement for chart aggregation."""
    if raw_agg is None:
        return "None / no explicit aggregation applied."

    agg_clean = str(raw_agg).strip().lower()
    if agg_clean in ("none", ""):
        return "None / no explicit aggregation applied."

    agg_upper = agg_clean.upper()

    # Ensure all signals are non-null strings
    str_signals = [str(s) for s in signals if s is not None]

    # 1. Check for explicit user request signal (Stage 8)
    for sig in str_signals:
        # e.g., "explicit MEDIAN aggregation applied to unit_price" or "explicit user request"
        m = re.search(r"explicit\s+([A-Za-z]+)\s+aggregation applied", sig, re.IGNORECASE)
        if m:
            detected_agg = m.group(1).upper()
            return f"User-requested {detected_agg} aggregation was applied."
        if "explicit user request" in sig.lower():
            return f"User-requested {agg_upper} aggregation was applied."

    # 2. Check for automatic / semantic selection signals (Stage 4 / 5 / 6)
    for sig in signals:
        sig_lower = sig.lower()
        if (
            "pipeline 1 semantic" in sig_lower
            or "conservative default" in sig_lower
            or "name tokens" in sig_lower
            or "aggregated using sum" in sig_lower
            or "aggregated using mean" in sig_lower
            or "resampled to" in sig_lower
        ):
            return "Semantic/default aggregation was applied."

    # If the candidate has a score, it originated from Stage 5 automatic ranking
    if has_score:
        return "Semantic/default aggregation was applied."

    # If aggregation is present but attribution cannot be established from signals
    return "Aggregation attribution is unavailable."


def _build_bar_evidence(points: list[dict[str, Any]]) -> list[str]:
    """Extract deterministic evidence for BAR and COUNT_BAR charts."""
    if not points:
        return []

    evidence: list[str] = [f"{len(points)} categories are displayed."]

    # Extract category labels and values
    parsed_points: list[tuple[str, float]] = []
    for p in points:
        x_val = str(p.get("x", ""))
        y_raw = p.get("y", 0)
        try:
            y_val = float(y_raw)
            parsed_points.append((x_val, y_val))
        except (TypeError, ValueError):
            continue

    if not parsed_points:
        return evidence

    values = [val for _, val in parsed_points]
    max_val = max(values)
    min_val = min(values)

    # Check if all values are equal
    if max_val == min_val:
        evidence.append(f"All displayed categories have equal value: {_fmt_num(max_val)}.")
        return evidence

    # Highest displayed categories
    highest = [x for x, val in parsed_points if val == max_val]
    if len(highest) == 1:
        evidence.append(f"Highest displayed value: {highest[0]} = {_fmt_num(max_val)}.")
    elif len(highest) == 2:
        evidence.append(
            f"Highest displayed value is tied: {highest[0]} = {_fmt_num(max_val)} and {highest[1]} = {_fmt_num(max_val)}."
        )
    else:
        joined = ", ".join(f"{cat} = {_fmt_num(max_val)}" for cat in highest)
        evidence.append(f"Highest displayed value is tied: {joined}.")

    # Lowest displayed categories
    lowest = [x for x, val in parsed_points if val == min_val]
    if len(lowest) == 1:
        evidence.append(f"Lowest displayed value: {lowest[0]} = {_fmt_num(min_val)}.")
    elif len(lowest) == 2:
        evidence.append(
            f"Lowest displayed value is tied: {lowest[0]} = {_fmt_num(min_val)} and {lowest[1]} = {_fmt_num(min_val)}."
        )
    else:
        joined = ", ".join(f"{cat} = {_fmt_num(min_val)}" for cat in lowest)
        evidence.append(f"Lowest displayed value is tied: {joined}.")

    return evidence


def _build_line_evidence(points: list[dict[str, Any]]) -> list[str]:
    """Extract deterministic evidence for LINE charts."""
    if not points:
        return []

    evidence: list[str] = []
    earliest = points[0]
    latest = points[-1]

    earliest_x = str(earliest.get("x", ""))
    earliest_y = earliest.get("y")
    latest_x = str(latest.get("x", ""))
    latest_y = latest.get("y")

    evidence.append(f"Earliest displayed point: {earliest_x} = {_fmt_num(earliest_y)}.")
    evidence.append(f"Latest displayed point: {latest_x} = {_fmt_num(latest_y)}.")

    try:
        ey = float(earliest_y)
        ly = float(latest_y)
        if ly > ey:
            evidence.append(
                f"The displayed endpoint value increased from {_fmt_num(ey)} to {_fmt_num(ly)}."
            )
        elif ly < ey:
            evidence.append(
                f"The displayed endpoint value decreased from {_fmt_num(ey)} to {_fmt_num(ly)}."
            )
        else:
            evidence.append(
                f"The displayed endpoint value stayed equal from {_fmt_num(ey)} to {_fmt_num(ly)}."
            )
    except (TypeError, ValueError):
        pass

    evidence.append(f"{len(points)} temporal points are displayed.")
    return evidence


def _build_scatter_evidence(points: list[dict[str, Any]], sampled: bool) -> list[str]:
    """Extract deterministic evidence for SCATTER charts without correlation claims."""
    if not points:
        return []

    evidence: list[str] = []
    n = len(points)
    if sampled:
        evidence.append(f"{n:,} paired observations are displayed after deterministic sampling.")
    else:
        evidence.append(f"{n:,} paired observations are displayed.")
        evidence.append("All available paired observations are represented.")

    return evidence


def _build_histogram_evidence(bins: list[dict[str, Any]]) -> list[str]:
    """Extract deterministic evidence for HISTOGRAM charts."""
    if not bins:
        return []

    evidence: list[str] = [f"{len(bins)} bins are displayed."]

    counts = [int(b.get("count", 0)) for b in bins]
    if not counts:
        return evidence

    max_c = max(counts)
    min_c = min(counts)

    # Highest frequency bin(s)
    highest_bins = [b for b in bins if int(b.get("count", 0)) == max_c]
    if len(highest_bins) == 1:
        hb = highest_bins[0]
        evidence.append(
            f"Highest-frequency bin: [{_fmt_num(hb.get('bin_start'))}, {_fmt_num(hb.get('bin_end'))}) = {max_c:,} observations."
        )
    elif len(highest_bins) == 2:
        hb1, hb2 = highest_bins[0], highest_bins[1]
        evidence.append(
            f"Highest-frequency bin is tied: [{_fmt_num(hb1.get('bin_start'))}, {_fmt_num(hb1.get('bin_end'))}) = {max_c:,} observations "
            f"and [{_fmt_num(hb2.get('bin_start'))}, {_fmt_num(hb2.get('bin_end'))}) = {max_c:,} observations."
        )
    else:
        evidence.append(f"Highest frequency is tied across {len(highest_bins)} bins ({max_c:,} observations).")

    # Lowest frequency bin(s)
    lowest_bins = [b for b in bins if int(b.get("count", 0)) == min_c]
    if len(lowest_bins) == 1:
        lb = lowest_bins[0]
        evidence.append(
            f"Lowest-frequency bin: [{_fmt_num(lb.get('bin_start'))}, {_fmt_num(lb.get('bin_end'))}) = {min_c:,} observations."
        )
    elif len(lowest_bins) == 2:
        lb1, lb2 = lowest_bins[0], lowest_bins[1]
        evidence.append(
            f"Lowest-frequency bin is tied: [{_fmt_num(lb1.get('bin_start'))}, {_fmt_num(lb1.get('bin_end'))}) = {min_c:,} observations "
            f"and [{_fmt_num(lb2.get('bin_start'))}, {_fmt_num(lb2.get('bin_end'))}) = {min_c:,} observations."
        )
    else:
        evidence.append(f"Lowest frequency is tied across {len(lowest_bins)} bins ({min_c:,} observations).")

    return evidence


def _validate_forbidden_language(text: str) -> None:
    """Verify that no forbidden causal, speculative, or subjective words appear in text."""
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    for w in words:
        if w in FORBIDDEN_WORDS:
            raise AssertionError(f"Forbidden term '{w}' found in explanation text: {text!r}")


def build_visualization_explanation(
    visualization: Mapping[str, Any],
    dataset_signals: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate a deterministic, factual explanation for a visualization payload item.

    Parameters
    ----------
    visualization:
        A visualization item mapping (produced by Stage 6 or Stage 8) or a top-level payload.
    dataset_signals:
        Optional sequence of dataset-level signals.

    Returns
    -------
    dict[str, Any]
        Deterministic JSON-safe explanation dictionary.
    """
    # -----------------------------------------------------------------------
    # 1. Resilience & Fallback Handling
    # -----------------------------------------------------------------------
    if not isinstance(visualization, Mapping):
        return _to_json_safe({
            "chart_type": "UNKNOWN",
            "x_column": "UNKNOWN",
            "y_column": None,
            "explanation": {
                "why_this_chart": "Visualization rationale is unavailable.",
                "what_it_shows": "No displayable chart data available.",
                "aggregation": "Aggregation attribution is unavailable.",
                "data_coverage": "No displayable chart data available.",
                "evidence": [],
            },
            "signals": ["Visualization data is unavailable."],
        })

    # If a top-level payload was passed (e.g. Stage 6/8 result with 'visualizations' list)
    target_item = visualization
    extracted_dataset_signals: list[str] = [
        str(s) for s in dataset_signals if s is not None
    ] if dataset_signals is not None else []
    if "visualizations" in visualization and isinstance(visualization["visualizations"], list):
        if "signals" in visualization and isinstance(visualization["signals"], list):
            extracted_dataset_signals.extend(str(s) for s in visualization["signals"] if s is not None)
        status = str(visualization.get("status", "")).upper()
        if status in ("INVALID", "NEEDS_CLARIFICATION", "NO_MEANINGFUL_VISUALIZATION") or not visualization["visualizations"]:
            return _to_json_safe({
                "chart_type": "UNKNOWN",
                "x_column": "UNKNOWN",
                "y_column": None,
                "explanation": {
                    "why_this_chart": "Visualization rationale is unavailable.",
                    "what_it_shows": "No displayable chart data available.",
                    "aggregation": "Aggregation attribution is unavailable.",
                    "data_coverage": "No displayable chart data available.",
                    "evidence": [],
                },
                "signals": ["Visualization data is unavailable."],
            })
        target_item = visualization["visualizations"][0]

    # Check for INVALID or NEEDS_CLARIFICATION status at item level
    item_status = str(target_item.get("status", "")).upper()
    if item_status in ("INVALID", "NEEDS_CLARIFICATION"):
        return _to_json_safe({
            "chart_type": str(target_item.get("chart_type")) if target_item.get("chart_type") else "UNKNOWN",
            "x_column": str(target_item.get("x_column")) if target_item.get("x_column") else "UNKNOWN",
            "y_column": str(target_item.get("y_column")) if target_item.get("y_column") is not None else None,
            "explanation": {
                "why_this_chart": "Visualization rationale is unavailable.",
                "what_it_shows": "No displayable chart data available.",
                "aggregation": "Aggregation attribution is unavailable.",
                "data_coverage": "No displayable chart data available.",
                "evidence": [],
            },
            "signals": ["Visualization data is unavailable."],
        })

    chart_type_raw = target_item.get("chart_type")
    x_col_raw = target_item.get("x_column")
    y_col_raw = target_item.get("y_column")
    chart_data = target_item.get("chart_data")

    # If chart_data is missing, None, or has_data is False
    if not isinstance(chart_data, Mapping) or not chart_data.get("has_data", True):
        return _to_json_safe({
            "chart_type": str(chart_type_raw) if chart_type_raw else "UNKNOWN",
            "x_column": str(x_col_raw) if x_col_raw else "UNKNOWN",
            "y_column": str(y_col_raw) if y_col_raw is not None else None,
            "explanation": {
                "why_this_chart": "Visualization rationale is unavailable.",
                "what_it_shows": "No displayable chart data available.",
                "aggregation": "Aggregation attribution is unavailable.",
                "data_coverage": "No displayable chart data available.",
                "evidence": [],
            },
            "signals": ["Visualization data is unavailable."],
        })

    chart_type = str(chart_data.get("chart_type") or chart_type_raw or "UNKNOWN")
    x_col = str(chart_data.get("x_column") or x_col_raw or "UNKNOWN")
    y_col = str(chart_data.get("y_column") or y_col_raw) if (chart_data.get("y_column") or y_col_raw) is not None else None

    points: list[dict[str, Any]] = list(chart_data.get("points") or [])
    bins: list[dict[str, Any]] = list(chart_data.get("bins") or [])
    usable_rows: int = int(chart_data.get("usable_row_count", 0))
    sampled: bool = bool(chart_data.get("sampled", False))
    raw_agg = chart_data.get("aggregation")

    # Collect signals for attribution analysis
    item_signals = [str(s) for s in target_item.get("signals", []) if isinstance(s, str)]
    cd_signals = [str(s) for s in chart_data.get("signals", []) if isinstance(s, str)]
    combined_signals = item_signals + cd_signals + extracted_dataset_signals

    has_score = target_item.get("score") is not None

    # Empty observations guard
    if not points and not bins:
        return _to_json_safe({
            "chart_type": chart_type,
            "x_column": x_col,
            "y_column": y_col,
            "explanation": {
                "why_this_chart": "Visualization rationale is unavailable.",
                "what_it_shows": "No displayable chart data available.",
                "aggregation": "Aggregation attribution is unavailable.",
                "data_coverage": "No displayable chart data available.",
                "evidence": [],
            },
            "signals": ["Visualization data is unavailable."],
        })

    # -----------------------------------------------------------------------
    # 2. Section 2: Chart Rationale (why_this_chart)
    # -----------------------------------------------------------------------
    presentation_variant = chart_data.get("presentation_variant")

    if chart_type == "COUNT_BAR":
        why_this_chart = f"Shows the distribution of observations across categories of '{x_col}'."
    elif chart_type == "BAR":
        why_this_chart = f"Shows a comparison of '{y_col}' across the categories of '{x_col}'."
    elif chart_type == "PIE":
        style = "doughnut" if presentation_variant == "DOUGHNUT" else "pie"
        why_this_chart = (
            f"Shows how the categories of '{x_col}' contribute to the whole as a {style} chart."
        )
    elif chart_type == "LINE":
        why_this_chart = f"Shows how '{y_col}' changes over the temporal dimension '{x_col}'."
    elif chart_type == "SCATTER":
        why_this_chart = f"Shows the relationship between the numeric measures '{x_col}' and '{y_col}'."
    elif chart_type == "HISTOGRAM":
        why_this_chart = f"Shows the distribution of values in the numeric measure '{x_col}'."
    else:
        why_this_chart = "Visualization rationale is unavailable."

    # -----------------------------------------------------------------------
    # 3. Section 3: What It Shows
    # -----------------------------------------------------------------------
    if chart_type == "COUNT_BAR":
        what_it_shows = f"Displays {len(points)} categories using observation counts."
    elif chart_type in ("BAR", "PIE"):
        if raw_agg is not None and str(raw_agg).lower() not in ("none", ""):
            agg_name = str(raw_agg).strip().upper()
            what_it_shows = f"Displays {len(points)} categories using {agg_name} aggregation."
        else:
            what_it_shows = f"Displays {len(points)} categories."
    elif chart_type == "LINE":
        what_it_shows = f"Displays {len(points)} temporal points in chronological order."
    elif chart_type == "SCATTER":
        if sampled:
            what_it_shows = f"Displays {len(points):,} paired observations after deterministic sampling."
        else:
            what_it_shows = f"Displays {len(points):,} paired observations."
    elif chart_type == "HISTOGRAM":
        total_obs = sum(int(b.get("count", 0)) for b in bins)
        what_it_shows = f"Displays {len(bins)} bins representing {total_obs:,} observations."
    else:
        what_it_shows = "No displayable chart data available."

    # -----------------------------------------------------------------------
    # 4. Section 4: Aggregation Attribution
    # -----------------------------------------------------------------------
    aggregation = _determine_aggregation_attribution(
        raw_agg=raw_agg,
        signals=combined_signals,
        has_score=has_score,
    )

    # -----------------------------------------------------------------------
    # 5. Section 5: Data Coverage
    # -----------------------------------------------------------------------
    if chart_type in ("BAR", "COUNT_BAR", "PIE"):
        if usable_rows > 0:
            coverage = f"{len(points)} displayed categories from {usable_rows:,} usable rows."
        else:
            coverage = f"{len(points)} displayed categories."
        if sampled:
            coverage += " Data was deterministically sampled to satisfy the visualization point limit."

    elif chart_type == "LINE":
        if usable_rows > 0:
            coverage = f"{len(points)} displayed temporal points from {usable_rows:,} usable rows."
        else:
            coverage = f"{len(points)} displayed temporal points."
        if sampled:
            coverage += " Data was deterministically sampled to satisfy the visualization point limit."

    elif chart_type == "SCATTER":
        coverage = f"{len(points):,} paired observations are displayed."
        if sampled:
            coverage += " Data was deterministically sampled to satisfy the visualization point limit."

    elif chart_type == "HISTOGRAM":
        coverage = f"{len(bins)} bins represent the available observations."
        if sampled:
            coverage += " Data was deterministically sampled to satisfy the visualization point limit."

    else:
        coverage = "No displayable chart data available."

    # -----------------------------------------------------------------------
    # 6. Section 6: Evidence
    # -----------------------------------------------------------------------
    if chart_type in ("BAR", "COUNT_BAR", "PIE"):
        evidence = _build_bar_evidence(points)
        item_signal = f"Evidence derived from {len(points)} displayed categories."
    elif chart_type == "LINE":
        evidence = _build_line_evidence(points)
        item_signal = f"Evidence derived from {len(points)} chronological points."
    elif chart_type == "SCATTER":
        evidence = _build_scatter_evidence(points, sampled=sampled)
        item_signal = f"Evidence derived from {len(points):,} paired observations."
    elif chart_type == "HISTOGRAM":
        evidence = _build_histogram_evidence(bins)
        item_signal = f"Evidence derived from {len(bins)} histogram bins."
    else:
        evidence = []
        item_signal = "Visualization data is unavailable."

    # Validate forbidden language
    _validate_forbidden_language(why_this_chart)
    _validate_forbidden_language(what_it_shows)
    _validate_forbidden_language(aggregation)
    _validate_forbidden_language(coverage)
    for ev in evidence:
        _validate_forbidden_language(ev)

    out_signals = [
        "Explanation generated from deterministic chart data.",
        item_signal,
    ]

    result: dict[str, Any] = {
        "chart_type": chart_type,
        "x_column": x_col,
        "y_column": y_col,
        "explanation": {
            "why_this_chart": why_this_chart,
            "what_it_shows": what_it_shows,
            "aggregation": aggregation,
            "data_coverage": coverage,
            "evidence": evidence,
        },
        "signals": out_signals,
    }

    return _to_json_safe(result)
