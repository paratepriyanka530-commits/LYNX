"""Results Service Composition Layer -- Pipeline 3, Stage 10.

Stage 10 of Pipeline 3: the application-facing composition layer for tabular
visualization intelligence.

Combines:
- Stage 6: Automatic visualization orchestration & contract
- Stage 7: Natural-language request parser & validator
- Stage 8: Requested visualization resolution
- Stage 9: Deterministic factual visualization explanations & XAI evidence

Execution Modes:
- Mode A (Automatic): When request_text is None or empty. Calls Stage 6 to
  generate candidate visualizations, enriches each with Stage 9 factual
  explanations, and returns a unified automatic payload.
- Mode B (Requested): When request_text is provided. Runs Stage 7 parser and
  validator. If VALID, calls Stage 8 resolver and enriches the resolved chart
  with Stage 9 explanation. If NEEDS_CLARIFICATION or INVALID, returns
  structured signals/options/errors without chart generation.

Invariants:
- Preserves Stage 6 candidate ranking and scores for automatic mode.
- Preserves score = None for user-requested mode (never generates scores).
- Preserves Stage 4/8 chart_data exactly.
- Enforces strict JSON safety with native Python primitives.
- Deterministic: identical inputs produce identical outputs with zero randomness.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from pipelines.pipeline3_results.visualization.column_classifier import (
    classify_dataframe,
)
from pipelines.pipeline3_results.visualization.explanation import (
    build_visualization_explanation,
)
from pipelines.pipeline3_results.visualization.feasibility import (
    analyze_visualization_feasibility,
)
from pipelines.pipeline3_results.visualization.integration import (
    _to_json_safe,
    build_visualization_payload,
)
from pipelines.pipeline3_results.visualization.request_parser import (
    parse_visualization_request,
)
from pipelines.pipeline3_results.visualization.request_resolver import (
    resolve_visualization_request,
)
from pipelines.pipeline3_results.visualization.request_validator import (
    validate_visualization_request,
)


def _enrich_with_explanation(
    item: Mapping[str, Any],
    dataset_signals: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Attach Stage 9 explanation sub-dict to a visualization item safely."""
    enriched = dict(item)
    try:
        explanation_res = build_visualization_explanation(
            visualization=item,
            dataset_signals=dataset_signals,
        )
        enriched["explanation"] = _to_json_safe(
            explanation_res.get("explanation", {})
        )
    except Exception:
        enriched["explanation"] = {
            "why_this_chart": "Visualization rationale is unavailable.",
            "what_it_shows": "No displayable chart data available.",
            "aggregation": "Aggregation attribution is unavailable.",
            "data_coverage": "No displayable chart data available.",
            "evidence": [],
        }
    return _to_json_safe(enriched)


def build_visualization_results(
    dataframe: pd.DataFrame,
    request_text: str | None = None,
    pipeline1_report: Mapping[str, Any] | None = None,
    dataset_name: str | None = None,
    max_visualizations: int = 5,
) -> dict[str, Any]:
    """Build unified Pipeline 3 visualization results.

    Parameters
    ----------
    dataframe:
        The cleaned / prepared pandas DataFrame to visualize.
    request_text:
        Optional natural-language visualization query (e.g., "Median price by category").
        If None or empty/whitespace-only, automatic mode is executed.
    pipeline1_report:
        Optional Pipeline 1 cleaning / profiling report mapping.
    dataset_name:
        Optional human-readable dataset name identifier.
    max_visualizations:
        Maximum number of visualizations to return for automatic mode (default: 5).
        Must be a positive integer (> 0).

    Returns
    -------
    dict[str, Any]
        Unified JSON-safe visualization response contract.

    Raises
    ------
    TypeError:
        If dataframe is not a pandas DataFrame, or request_text / dataset_name /
        pipeline1_report have invalid types.
    ValueError:
        If max_visualizations is not a positive integer (> 0) or is a boolean.
    """
    # -----------------------------------------------------------------------
    # 1. Input Validation
    # -----------------------------------------------------------------------
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"dataframe must be a pandas DataFrame, got {type(dataframe).__name__}"
        )

    if request_text is not None and not isinstance(request_text, str):
        raise TypeError(
            f"request_text must be a string or None, got {type(request_text).__name__}"
        )

    if pipeline1_report is not None and not isinstance(pipeline1_report, Mapping):
        raise TypeError(
            f"pipeline1_report must be a Mapping or None, got {type(pipeline1_report).__name__}"
        )

    if dataset_name is not None and (not isinstance(dataset_name, str) or isinstance(dataset_name, bool)):
        raise TypeError(
            f"dataset_name must be a string or None, got {type(dataset_name).__name__}"
        )

    if isinstance(max_visualizations, bool) or not isinstance(max_visualizations, int) or max_visualizations <= 0:
        raise ValueError(
            f"max_visualizations must be a positive integer (> 0), got {max_visualizations!r}"
        )

    clean_dataset_name = str(dataset_name) if dataset_name is not None else None

    # -----------------------------------------------------------------------
    # 2. Execution Routing: Mode A (Automatic) vs Mode B (Requested)
    # -----------------------------------------------------------------------
    is_requested = request_text is not None and bool(request_text.strip())

    if not is_requested:
        # -------------------------------------------------------------------
        # MODE A: Automatic Visualization
        # -------------------------------------------------------------------
        payload = build_visualization_payload(
            dataframe=dataframe,
            pipeline1_report=pipeline1_report,
            dataset_name=clean_dataset_name,
            max_visualizations=max_visualizations,
        )

        top_signals = [str(s) for s in payload.get("signals", []) if s is not None]
        enriched_visualizations = [
            _enrich_with_explanation(v, dataset_signals=top_signals)
            for v in payload.get("visualizations", [])
        ]

        result: dict[str, Any] = {
            "mode": "automatic",
            "status": str(payload.get("status", "NO_MEANINGFUL_VISUALIZATION")),
            "dataset_name": clean_dataset_name,
            "visualization_count": len(enriched_visualizations),
            "visualizations": enriched_visualizations,
            "signals": top_signals,
        }
        return _to_json_safe(result)

    # -----------------------------------------------------------------------
    # MODE B: User-Requested Visualization
    # -----------------------------------------------------------------------
    assert request_text is not None
    clean_request_text = request_text.strip()

    # Guard: empty dataframe for requested mode
    if len(dataframe) == 0:
        return _to_json_safe({
            "mode": "requested",
            "request_text": clean_request_text,
            "status": "NO_MEANINGFUL_VISUALIZATION",
            "dataset_name": clean_dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "signals": ["dataframe has zero rows -- no chart data can be produced"],
        })

    # Step 1 & 2: Classify and Analyze Feasibility
    classification = classify_dataframe(
        dataframe=dataframe,
        pipeline1_report=pipeline1_report,
        dataset_name=clean_dataset_name,
    )
    feasibility = analyze_visualization_feasibility(
        dataframe=dataframe,
        classification=classification,
    )

    # Step 3: Parse Request
    parsed = parse_visualization_request(
        request_text=clean_request_text,
        dataframe=dataframe,
        classification=classification,
    )

    # Step 4: Validate Request
    validated = validate_visualization_request(
        dataframe=dataframe,
        classification=classification,
        request=parsed,
        feasibility=feasibility,
    )

    # Step 5: Check Validation Status
    if validated.status == "NEEDS_CLARIFICATION":
        clarification_res: dict[str, Any] = {
            "mode": "requested",
            "request_text": clean_request_text,
            "status": "NEEDS_CLARIFICATION",
            "dataset_name": clean_dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "clarification_options": [str(opt) for opt in validated.clarification_options],
            "signals": [str(s) for s in validated.signals],
        }
        return _to_json_safe(clarification_res)

    if validated.status == "INVALID":
        invalid_res: dict[str, Any] = {
            "mode": "requested",
            "request_text": clean_request_text,
            "status": "INVALID",
            "dataset_name": clean_dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "errors": [str(e) for e in validated.errors],
            "signals": [str(s) for s in validated.signals],
        }
        return _to_json_safe(invalid_res)

    # Step 6: Resolve Valid Request via Stage 8
    resolved = resolve_visualization_request(
        dataframe=dataframe,
        validated_request=validated,
        pipeline1_report=pipeline1_report,
        dataset_name=clean_dataset_name,
    )

    resolved_status = str(resolved.get("status", "INVALID"))
    resolved_signals = [str(s) for s in resolved.get("signals", []) if s is not None]

    if resolved_status == "READY":
        enriched_visualizations = [
            _enrich_with_explanation(v, dataset_signals=resolved_signals)
            for v in resolved.get("visualizations", [])
        ]
        ready_res: dict[str, Any] = {
            "mode": "requested",
            "request_text": clean_request_text,
            "status": "READY",
            "dataset_name": clean_dataset_name,
            "visualization_count": len(enriched_visualizations),
            "visualizations": enriched_visualizations,
            "signals": resolved_signals,
        }
        return _to_json_safe(ready_res)

    if resolved_status == "NEEDS_CLARIFICATION":
        nc_res: dict[str, Any] = {
            "mode": "requested",
            "request_text": clean_request_text,
            "status": "NEEDS_CLARIFICATION",
            "dataset_name": clean_dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "clarification_options": [str(opt) for opt in resolved.get("clarification_options", [])],
            "signals": resolved_signals,
        }
        return _to_json_safe(nc_res)

    if resolved_status == "NO_MEANINGFUL_VISUALIZATION":
        nmv_res: dict[str, Any] = {
            "mode": "requested",
            "request_text": clean_request_text,
            "status": "NO_MEANINGFUL_VISUALIZATION",
            "dataset_name": clean_dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "signals": resolved_signals,
        }
        return _to_json_safe(nmv_res)

    # Fallback to INVALID for any other failure
    fallback_invalid: dict[str, Any] = {
        "mode": "requested",
        "request_text": clean_request_text,
        "status": "INVALID",
        "dataset_name": clean_dataset_name,
        "visualization_count": 0,
        "visualizations": [],
        "errors": [str(e) for e in resolved.get("errors", ["Visualization request could not be resolved."])],
        "signals": resolved_signals,
    }
    return _to_json_safe(fallback_invalid)