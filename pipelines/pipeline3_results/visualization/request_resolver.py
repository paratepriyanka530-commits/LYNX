"""Deterministic Requested Visualization Resolution — Pipeline 3, Stage 8.

Stage 8 of Pipeline 3: executes a validated user visualization request
(ValidatedVisualizationRequest) produced by Stage 7.

Architecture:
    User request
    → Stage 7 parser (parse_visualization_request)
    → Stage 7 validator (validate_visualization_request)
    → ValidatedVisualizationRequest
    → Stage 8 resolver (resolve_visualization_request)
    → Stage 4 deterministic chart-data generation
    → Frontend-safe requested visualization payload

Design Principles:
* Pure execution layer: Stage 8 only executes already validated user requests.
* Explicit user aggregations (SUM, MEAN, MEDIAN, MIN, MAX, COUNT) are honored directly
  for compatible chart types (BAR, LINE) and never silently replaced by defaults.
* Incompatible chart/aggregation combinations return factual INVALID responses.
* score is ALWAYS None (user requests do not have ranking or usefulness scores).
* Pure JSON-safe native Python dictionary returned (no external dataclass/model objects).
* Does not render charts, manipulate UI, invoke APIs, or call ML/LLM services.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pipelines.pipeline3_results.visualization.column_classifier import (
    classify_dataframe,
)
from pipelines.pipeline3_results.visualization.data_builder import (
    STAGE4_MAX_CATEGORY_POINTS,
    STAGE4_MAX_LINE_POINTS,
    _build_count_bar,
    _build_histogram,
    _build_scatter,
    _choose_temporal_granularity,
    _safe_float,
    _select_aggregation,
    _to_py_scalar,
)
from pipelines.pipeline3_results.visualization.integration import _to_json_safe
from pipelines.pipeline3_results.visualization.schemas import (
    ChartData,
    ChartPoint,
    DatasetClassification,
    RankedCandidate,
    ValidatedVisualizationRequest,
)


def _build_explicit_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    agg_str: str,
) -> ChartData:
    """Build BAR chart data with an explicit user aggregation function."""
    signals: list[str] = []
    valid = df[[x_col, y_col]].copy()
    valid[y_col] = pd.to_numeric(valid[y_col], errors="coerce")
    initial_rows = len(valid)
    valid = valid.dropna(subset=[x_col, y_col])

    if len(valid) == 0:
        return ChartData(
            chart_type="BAR",
            x_column=x_col,
            y_column=y_col,
            aggregation=agg_str.lower(),  # type: ignore[arg-type]
            points=[],
            bins=[],
            signals=[f"0 usable rows after filtering for BAR({x_col}, {y_col})"],
            usable_row_count=0,
            sampled=False,
            has_data=False,
        )

    valid = valid[valid[y_col].apply(lambda v: math.isfinite(float(v)))]
    valid = valid[valid[x_col].astype(str).str.strip() != ""]
    usable = len(valid)
    dropped = initial_rows - usable
    if dropped:
        signals.append(f"{dropped} rows excluded (missing or non-finite values)")
    signals.append(f"{usable} valid observations used")

    if usable == 0:
        return ChartData(
            chart_type="BAR",
            x_column=x_col,
            y_column=y_col,
            aggregation=agg_str.lower(),  # type: ignore[arg-type]
            points=[],
            bins=[],
            signals=signals,
            usable_row_count=0,
            sampled=False,
            has_data=False,
        )

    signals.append(f"explicit {agg_str} aggregation applied to {y_col}")

    grouped = valid.groupby(x_col, sort=False)[y_col]
    agg_lower = agg_str.lower()
    if agg_lower == "sum":
        agg_series = grouped.sum()
    elif agg_lower == "mean":
        agg_series = grouped.mean()
    elif agg_lower == "median":
        agg_series = grouped.median()
    elif agg_lower == "min":
        agg_series = grouped.min()
    elif agg_lower == "max":
        agg_series = grouped.max()
    elif agg_lower == "count":
        agg_series = grouped.count()
    else:
        agg_series = grouped.mean()

    agg_df = agg_series.reset_index()
    agg_df.columns = ["x", "y"]
    agg_df = agg_df.dropna(subset=["y"])
    agg_df = agg_df[agg_df["y"].apply(lambda v: math.isfinite(float(v)))]
    agg_df = agg_df.sort_values(
        by=["y", "x"], ascending=[False, True]
    ).reset_index(drop=True)

    sampled = False
    total_groups = len(agg_df)
    if total_groups > STAGE4_MAX_CATEGORY_POINTS:
        agg_df = agg_df.iloc[:STAGE4_MAX_CATEGORY_POINTS]
        signals.append(
            f"truncated to top {STAGE4_MAX_CATEGORY_POINTS} categories "
            f"(from {total_groups}) by {agg_lower} value"
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
        aggregation=agg_lower,  # type: ignore[arg-type]
        points=points,
        bins=[],
        signals=signals,
        usable_row_count=usable,
        sampled=sampled,
        has_data=len(points) > 0,
    )


def _build_default_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    classification: DatasetClassification,
) -> ChartData:
    """Build BAR chart data using Stage 4 semantic aggregation selection."""
    signals: list[str] = []
    valid = df[[x_col, y_col]].copy()
    valid[y_col] = pd.to_numeric(valid[y_col], errors="coerce")
    initial_rows = len(valid)
    valid = valid.dropna(subset=[x_col, y_col])

    if len(valid) == 0:
        return ChartData(
            chart_type="BAR",
            x_column=x_col,
            y_column=y_col,
            aggregation=None,
            points=[],
            bins=[],
            signals=[f"0 usable rows after filtering for BAR({x_col}, {y_col})"],
            usable_row_count=0,
            sampled=False,
            has_data=False,
        )

    valid = valid[valid[y_col].apply(lambda v: math.isfinite(float(v)))]
    valid = valid[valid[x_col].astype(str).str.strip() != ""]
    usable = len(valid)
    dropped = initial_rows - usable
    if dropped:
        signals.append(f"{dropped} rows excluded (missing or non-finite values)")
    signals.append(f"{usable} valid observations used")

    if usable == 0:
        return ChartData(
            chart_type="BAR",
            x_column=x_col,
            y_column=y_col,
            aggregation=None,
            points=[],
            bins=[],
            signals=signals,
            usable_row_count=0,
            sampled=False,
            has_data=False,
        )

    agg, agg_signal = _select_aggregation(y_col, classification)
    signals.append(agg_signal)

    grouped = valid.groupby(x_col, sort=False)[y_col]
    if agg == "sum":
        agg_series = grouped.sum()
    else:
        agg_series = grouped.mean()

    agg_df = agg_series.reset_index()
    agg_df.columns = ["x", "y"]
    agg_df = agg_df.dropna(subset=["y"])
    agg_df = agg_df[agg_df["y"].apply(lambda v: math.isfinite(float(v)))]
    agg_df = agg_df.sort_values(
        by=["y", "x"], ascending=[False, True]
    ).reset_index(drop=True)

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
        has_data=len(points) > 0,
    )


def _build_explicit_line(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    agg_str: str,
) -> ChartData:
    """Build LINE chart data with an explicit user aggregation function."""
    signals: list[str] = []
    working = df[[x_col, y_col]].copy()
    working[y_col] = pd.to_numeric(working[y_col], errors="coerce")

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

    agg_lower = agg_str.lower()
    if usable == 0:
        return ChartData(
            chart_type="LINE",
            x_column=x_col,
            y_column=y_col,
            aggregation=agg_lower,  # type: ignore[arg-type]
            points=[],
            bins=[],
            signals=signals,
            usable_row_count=0,
            sampled=False,
            has_data=False,
        )

    signals.append(f"explicit {agg_str} aggregation applied to {y_col}")

    freq, freq_label = _choose_temporal_granularity(working["_x_parsed"])
    signals.append(f"temporal granularity: {freq_label}")

    if freq:
        ts = working.set_index("_x_parsed")[y_col]
        if agg_lower == "sum":
            resampled = ts.resample(freq).sum()
        elif agg_lower == "mean":
            resampled = ts.resample(freq).mean()
        elif agg_lower == "median":
            resampled = ts.resample(freq).median()
        elif agg_lower == "min":
            resampled = ts.resample(freq).min()
        elif agg_lower == "max":
            resampled = ts.resample(freq).max()
        elif agg_lower == "count":
            resampled = ts.resample(freq).count()
        else:
            resampled = ts.resample(freq).mean()
        resampled = resampled.dropna()
        agg_df = resampled.reset_index()
        agg_df.columns = ["_x_parsed", y_col]
        signals.append(f"resampled to {freq_label} ({len(agg_df)} periods)")
    else:
        working_sorted = working.sort_values("_x_parsed").reset_index(drop=True)
        agg_df = working_sorted[["_x_parsed", y_col]].copy()
        signals.append(f"no resampling applied ({len(agg_df)} raw periods)")

    agg_df = agg_df.sort_values("_x_parsed").reset_index(drop=True)

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
        aggregation=agg_lower,  # type: ignore[arg-type]
        points=points,
        bins=[],
        signals=signals,
        usable_row_count=usable,
        sampled=sampled,
        has_data=len(points) > 0,
    )


def _build_default_line(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    classification: DatasetClassification,
) -> ChartData:
    """Build LINE chart data using Stage 4 semantic aggregation selection."""
    signals: list[str] = []
    working = df[[x_col, y_col]].copy()
    working[y_col] = pd.to_numeric(working[y_col], errors="coerce")

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
        return ChartData(
            chart_type="LINE",
            x_column=x_col,
            y_column=y_col,
            aggregation=None,
            points=[],
            bins=[],
            signals=signals,
            usable_row_count=0,
            sampled=False,
            has_data=False,
        )

    agg, agg_signal = _select_aggregation(y_col, classification)
    signals.append(agg_signal)

    freq, freq_label = _choose_temporal_granularity(working["_x_parsed"])
    signals.append(f"temporal granularity: {freq_label}")

    if freq:
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
        working_sorted = working.sort_values("_x_parsed").reset_index(drop=True)
        agg_df = working_sorted[["_x_parsed", y_col]].copy()
        signals.append(f"no resampling applied ({len(agg_df)} raw periods)")

    agg_df = agg_df.sort_values("_x_parsed").reset_index(drop=True)

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


def resolve_visualization_request(
    dataframe: pd.DataFrame,
    validated_request: ValidatedVisualizationRequest,
    pipeline1_report: Mapping[str, Any] | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Execute a validated user visualization request into a frontend-safe payload.

    Parameters
    ----------
    dataframe:
        The target cleaned/prepared pandas DataFrame.
    validated_request:
        The validated visualization request produced by Stage 7.
    pipeline1_report:
        Optional Pipeline 1 cleaning / profiling report mapping.
    dataset_name:
        Optional dataset name string identifier.

    Returns
    -------
    dict[str, Any]
        Deterministic, JSON-serializable dictionary payload with status
        READY, NEEDS_CLARIFICATION, INVALID, or NO_MEANINGFUL_VISUALIZATION.

    Raises
    ------
    TypeError
        If inputs do not match their expected types.
    """
    # -----------------------------------------------------------------------
    # 1. Strict Input Type Validation
    # -----------------------------------------------------------------------
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"dataframe must be a pandas DataFrame, got {type(dataframe).__name__}")
    if not isinstance(validated_request, ValidatedVisualizationRequest):
        raise TypeError(
            f"validated_request must be a ValidatedVisualizationRequest, got {type(validated_request).__name__}"
        )
    if pipeline1_report is not None and not isinstance(pipeline1_report, Mapping):
        raise TypeError(
            f"pipeline1_report must be a Mapping or None, got {type(pipeline1_report).__name__}"
        )
    if dataset_name is not None and (not isinstance(dataset_name, str) or isinstance(dataset_name, bool)):
        raise TypeError(
            f"dataset_name must be a string or None, got {type(dataset_name).__name__}"
        )

    # -----------------------------------------------------------------------
    # 2. Non-Valid Request States
    # -----------------------------------------------------------------------
    if validated_request.status == "NEEDS_CLARIFICATION":
        return _to_json_safe({
            "status": "NEEDS_CLARIFICATION",
            "dataset_name": dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "clarification_options": [str(opt) for opt in validated_request.clarification_options],
            "signals": [str(s) for s in validated_request.signals],
        })

    if validated_request.status == "INVALID":
        return _to_json_safe({
            "status": "INVALID",
            "dataset_name": dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "errors": [str(e) for e in validated_request.errors],
            "signals": [str(s) for s in validated_request.signals],
        })

    # -----------------------------------------------------------------------
    # 3. Valid Request Guards: Column Existence & Aggregation Compatibility
    # -----------------------------------------------------------------------
    x_col = validated_request.x_column
    y_col = validated_request.y_column
    chart_type = validated_request.chart_type
    agg = validated_request.aggregation

    # Verify column existence in dataframe
    missing_cols: list[str] = []
    if x_col is not None and x_col not in dataframe.columns:
        missing_cols.append(x_col)
    if y_col is not None and y_col not in dataframe.columns:
        missing_cols.append(y_col)

    if missing_cols:
        return _to_json_safe({
            "status": "INVALID",
            "dataset_name": dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "errors": [f"Requested column(s) {missing_cols} do not exist in dataset."],
            "signals": list(validated_request.signals) + [f"rejected: missing column(s) {missing_cols}"],
        })

    # Aggregation compatibility enforcement
    if chart_type == "SCATTER":
        is_subtoken_agg = any("aggregation derived from column token:" in s for s in validated_request.signals) and not any(
            "aggregation detected:" in s for s in validated_request.signals
        )
        if agg is not None and not is_subtoken_agg:
            return _to_json_safe({
                "status": "INVALID",
                "dataset_name": dataset_name,
                "visualization_count": 0,
                "visualizations": [],
                "errors": [
                    f"SCATTER plot does not support aggregation '{agg}'. "
                    "Scatter plots display individual unaggregated data points."
                ],
                "signals": list(validated_request.signals) + [
                    f"rejected: SCATTER does not support aggregation '{agg}'"
                ],
            })

    elif chart_type == "HISTOGRAM":
        if agg is not None and agg != "COUNT":
            return _to_json_safe({
                "status": "INVALID",
                "dataset_name": dataset_name,
                "visualization_count": 0,
                "visualizations": [],
                "errors": [
                    f"HISTOGRAM chart does not support aggregation '{agg}'. "
                    "Histograms display numeric distribution frequencies (bin counts)."
                ],
                "signals": list(validated_request.signals) + [
                    f"rejected: HISTOGRAM does not support aggregation '{agg}'"
                ],
            })

    elif chart_type == "COUNT_BAR":
        if agg is not None and agg != "COUNT":
            return _to_json_safe({
                "status": "INVALID",
                "dataset_name": dataset_name,
                "visualization_count": 0,
                "visualizations": [],
                "errors": [
                    f"COUNT_BAR chart does not support aggregation '{agg}'. "
                    "COUNT_BAR charts display category observation counts."
                ],
                "signals": list(validated_request.signals) + [
                    f"rejected: COUNT_BAR does not support aggregation '{agg}'"
                ],
            })

    elif chart_type == "BAR":
        if agg is not None and agg not in ("SUM", "MEAN", "MEDIAN", "MIN", "MAX", "COUNT"):
            return _to_json_safe({
                "status": "INVALID",
                "dataset_name": dataset_name,
                "visualization_count": 0,
                "visualizations": [],
                "errors": [
                    f"BAR chart does not support aggregation '{agg}'. "
                    "Supported aggregations: SUM, MEAN, MEDIAN, MIN, MAX, COUNT."
                ],
                "signals": list(validated_request.signals) + [
                    f"rejected: BAR does not support aggregation '{agg}'"
                ],
            })

    elif chart_type == "LINE":
        if agg is not None and agg not in ("SUM", "MEAN", "MEDIAN", "MIN", "MAX", "COUNT"):
            return _to_json_safe({
                "status": "INVALID",
                "dataset_name": dataset_name,
                "visualization_count": 0,
                "visualizations": [],
                "errors": [
                    f"LINE chart does not support aggregation '{agg}'. "
                    "Supported aggregations: SUM, MEAN, MEDIAN, MIN, MAX, COUNT."
                ],
                "signals": list(validated_request.signals) + [
                    f"rejected: LINE does not support aggregation '{agg}'"
                ],
            })

    # Guard: empty dataframe
    if len(dataframe) == 0:
        return _to_json_safe({
            "status": "NO_MEANINGFUL_VISUALIZATION",
            "dataset_name": dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "signals": list(validated_request.signals) + ["dataframe has zero rows — no chart data can be produced"],
        })

    # -----------------------------------------------------------------------
    # 4. Chart Data Generation
    # -----------------------------------------------------------------------
    candidate = RankedCandidate(
        chart_type=chart_type,  # type: ignore[arg-type]
        x_column=x_col or "",
        y_column=y_col,
        score=0.0,
        rank=1,
    )

    if chart_type == "COUNT_BAR":
        chart_data = _build_count_bar(dataframe, candidate)

    elif chart_type == "HISTOGRAM":
        chart_data = _build_histogram(dataframe, candidate)

    elif chart_type == "SCATTER":
        chart_data = _build_scatter(dataframe, candidate)

    elif chart_type == "BAR":
        if agg is None:
            classification = classify_dataframe(
                dataframe=dataframe,
                pipeline1_report=pipeline1_report,
                dataset_name=dataset_name,
            )
            chart_data = _build_default_bar(
                df=dataframe,
                x_col=x_col,  # type: ignore[arg-type]
                y_col=y_col,  # type: ignore[arg-type]
                classification=classification,
            )
        else:
            chart_data = _build_explicit_bar(
                df=dataframe,
                x_col=x_col,  # type: ignore[arg-type]
                y_col=y_col,  # type: ignore[arg-type]
                agg_str=agg,
            )

    elif chart_type == "LINE":
        if agg is None:
            classification = classify_dataframe(
                dataframe=dataframe,
                pipeline1_report=pipeline1_report,
                dataset_name=dataset_name,
            )
            chart_data = _build_default_line(
                df=dataframe,
                x_col=x_col,  # type: ignore[arg-type]
                y_col=y_col,  # type: ignore[arg-type]
                classification=classification,
            )
        else:
            chart_data = _build_explicit_line(
                df=dataframe,
                x_col=x_col,  # type: ignore[arg-type]
                y_col=y_col,  # type: ignore[arg-type]
                agg_str=agg,
            )

    else:
        return _to_json_safe({
            "status": "INVALID",
            "dataset_name": dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "errors": [f"Unsupported chart type '{chart_type}'."],
            "signals": list(validated_request.signals) + [f"rejected: unsupported chart type '{chart_type}'"],
        })

    # -----------------------------------------------------------------------
    # 5. No-Data Guard
    # -----------------------------------------------------------------------
    if chart_data.usable_row_count == 0 or not chart_data.has_data:
        return _to_json_safe({
            "status": "NO_MEANINGFUL_VISUALIZATION",
            "dataset_name": dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "signals": list(validated_request.signals) + list(chart_data.signals) + [
                "no usable observations produced for requested visualization"
            ],
        })

    # -----------------------------------------------------------------------
    # 6. Final Payload Construction (score MUST ALWAYS be None)
    # -----------------------------------------------------------------------
    chart_data_dict = _to_json_safe(chart_data.model_dump(mode="json"))

    visualization_item: dict[str, Any] = {
        "rank": 1,
        "chart_type": str(chart_data.chart_type),
        "x_column": str(chart_data.x_column),
        "y_column": str(chart_data.y_column) if chart_data.y_column is not None else None,
        "score": None,  # Crucial: score must always be None for user-requested visualizations
        "chart_data": chart_data_dict,
        "signals": [str(s) for s in chart_data.signals],
    }

    top_signals = list(validated_request.signals)
    resolution_signal = f"resolved requested visualization: {chart_type}(x={x_col}, y={y_col})"
    if resolution_signal not in top_signals:
        top_signals.append(resolution_signal)

    payload: dict[str, Any] = {
        "status": "READY",
        "dataset_name": dataset_name,
        "visualization_count": 1,
        "visualizations": [visualization_item],
        "signals": [str(s) for s in top_signals],
    }

    return _to_json_safe(payload)
