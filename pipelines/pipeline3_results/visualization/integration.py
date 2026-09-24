"""Visualization Integration Contract — Pipeline 3, Stage 6.

Stage 6 of Pipeline 3: converts the Stage 5 VisualizationResult into a stable,
deterministic, frontend-safe JSON-serializable contract dictionary.

Design Principles:
* Pure dictionary/list contract (no internal Pydantic/pandas/numpy classes exposed).
* Strictly deterministic: identical inputs produce identical dictionary structures.
* Zero modification of Stage 1–5 ranking, scoring, classification, or chart data.
* Full preservation of factual signals and candidate ordering.
* Does not render charts, manipulate UI, invoke APIs, or call ML/LLM services.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pipelines.pipeline3_results.visualization.orchestrator import (
    build_visualization_result,
)
from pipelines.pipeline3_results.visualization.schemas import VisualizationResult


def _to_json_safe(val: Any) -> Any:
    """Recursively convert values to native Python JSON-serializable primitives."""
    if val is None:
        return None
    if isinstance(val, (bool, str)):
        return val
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        if pd.isna(val) or not math.isfinite(val):
            return None
        return float(val)
    if isinstance(val, dict):
        return {str(k): _to_json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_to_json_safe(v) for v in val]
    if hasattr(val, "model_dump"):
        return _to_json_safe(val.model_dump(mode="json"))
    return str(val)


def build_visualization_payload(
    dataframe: pd.DataFrame,
    pipeline1_report: Mapping[str, Any] | None = None,
    dataset_name: str | None = None,
    max_visualizations: int = 5,
) -> dict[str, Any]:
    """Build a deterministic, frontend-safe JSON-serializable visualization payload.

    Converts the output of Stage 5 (orchestration) into a stable integration contract
    composed entirely of standard Python primitives (dict, list, str, int, float, bool, None).

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
    dict[str, Any]
        Deterministic JSON-safe payload with:
        - status: "READY" | "NO_MEANINGFUL_VISUALIZATION"
        - dataset_name: str | None
        - visualization_count: int
        - visualizations: list[dict[str, Any]]
        - signals: list[str]

    Raises
    ------
    TypeError
        If dataframe is not a pandas DataFrame.
    ValueError
        If max_visualizations is not a positive integer (> 0) or is a bool.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"dataframe must be a pandas DataFrame, got {type(dataframe).__name__}")

    if isinstance(max_visualizations, bool) or not isinstance(max_visualizations, int) or max_visualizations <= 0:
        raise ValueError(
            f"max_visualizations must be a positive integer (> 0), got {max_visualizations!r}"
        )

    # Call Stage 5 orchestrator
    result: VisualizationResult = build_visualization_result(
        dataframe=dataframe,
        pipeline1_report=pipeline1_report,
        dataset_name=dataset_name,
        max_visualizations=max_visualizations,
    )

    visualizations_list: list[dict[str, Any]] = []
    for item in result.visualizations:
        chart_data_dict = _to_json_safe(item.chart_data.model_dump(mode="json"))
        visualizations_list.append({
            "rank": int(item.rank),
            "chart_type": str(item.chart_type),
            "x_column": str(item.x_column),
            "y_column": str(item.y_column) if item.y_column is not None else None,
            "score": float(item.score),
            "chart_data": chart_data_dict,
            "signals": [str(s) for s in item.signals],
        })

    payload: dict[str, Any] = {
        "status": str(result.status),
        "dataset_name": str(result.dataset_name) if result.dataset_name is not None else None,
        "visualization_count": int(result.visualization_count),
        "visualizations": visualizations_list,
        "signals": [str(s) for s in result.signals],
    }

    return payload

