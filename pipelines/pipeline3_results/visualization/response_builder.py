"""Unified Visualization Response Contract -- Pipeline 3, Stage 10.

Stage 10 of Pipeline 3: assembles already-generated Stage 6 (automatic) or
Stage 8 (requested) visualization payloads together with Stage 9 deterministic
factual explanations into one stable, frontend-safe, JSON-serializable response
contract.

Design Principles:
* Assembler Only: never reranks, rescores, or reorders visualizations.
* Score Invariance: the incoming score value (float for automatic, None for
  requested) is preserved exactly; Stage 10 never generates a score.
* Non-Mutating: the input payload is never altered in-place.
* Pure JSON Safety: returns only standard Python primitives.
* Deterministic: identical inputs produce identical outputs.
* Mode Validation: rejects any mode other than "automatic" or "requested".
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from pipelines.pipeline3_results.visualization.explanation import (
    build_visualization_explanation,
)
from pipelines.pipeline3_results.visualization.integration import _to_json_safe

# Public constant: accepted mode values
SUPPORTED_MODES: frozenset[str] = frozenset({"automatic", "requested"})

# Standard keys shared by Stage 6 and Stage 8 top-level payloads
_STANDARD_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {"status", "dataset_name", "visualization_count", "visualizations", "signals"}
)


def _enrich_visualization_item(
    viz_item: Mapping[str, Any],
    dataset_signals: list[str] | None,
) -> dict[str, Any]:
    """Return a shallow copy of viz_item with a Stage 9 explanation attached.

    The explanation is derived by calling build_visualization_explanation from
    Stage 9.  The returned dict is JSON-safe.  The input is not mutated.
    """
    # Call Stage 9 -- pass the single visualization item (not a full payload)
    explanation_result = build_visualization_explanation(
        visualization=viz_item,
        dataset_signals=dataset_signals,
    )

    # Copy all fields from the original item, then attach the explanation sub-dict
    enriched: dict[str, Any] = {}
    for key, value in viz_item.items():
        enriched[key] = _to_json_safe(value)

    # Attach only the inner explanation mapping (not the Stage 9 wrapper)
    enriched["explanation"] = _to_json_safe(
        explanation_result.get("explanation", {})
    )
    return enriched


def build_visualization_response(
    visualization_payload: Mapping[str, Any],
    mode: str,
    dataset_signals: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a unified, frontend-safe Stage 10 visualization response.

    Accepts a Stage 6 automatic payload or a Stage 8 requested payload and
    attaches Stage 9 deterministic factual explanations to every visualization
    item, returning a single JSON-serializable response object.

    Parameters
    ----------
    visualization_payload:
        A dict produced by build_visualization_payload (Stage 6) or
        resolve_visualization_request (Stage 8).  Must be a Mapping.
    mode:
        Either "automatic" or "requested".  Any other value raises ValueError.
    dataset_signals:
        Optional sequence of dataset-level signal strings forwarded to Stage 9.

    Returns
    -------
    dict[str, Any]
        JSON-safe response contract:
        {
            "mode": "automatic" | "requested",
            "status": str,
            "dataset_name": str | None,
            "visualization_count": int,
            "visualizations": [
                {
                    "rank": int,
                    "chart_type": str,
                    "x_column": str,
                    "y_column": str | None,
                    "score": float | None,
                    "chart_data": dict,
                    "signals": list[str],
                    "explanation": {
                        "why_this_chart": str,
                        "what_it_shows": str,
                        "aggregation": str,
                        "data_coverage": str,
                        "evidence": list[str],
                    },
                },
                ...
            ],
            "signals": list[str],
            # plus any extra Stage 8 fields (errors, clarification_options)
        }

    Raises
    ------
    TypeError
        If visualization_payload is not a Mapping.
    ValueError
        If mode is not "automatic" or "requested".
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    if not isinstance(visualization_payload, Mapping):
        raise TypeError(
            "visualization_payload must be a Mapping, got "
            f"{type(visualization_payload).__name__}"
        )
    if not isinstance(mode, str) or mode not in SUPPORTED_MODES:
        raise ValueError(
            f"mode must be 'automatic' or 'requested', got {mode!r}"
        )

    # Normalise dataset_signals to a plain list of strings (or None)
    safe_signals: list[str] | None = None
    if dataset_signals is not None:
        safe_signals = [str(s) for s in dataset_signals if s is not None]

    # ------------------------------------------------------------------
    # 2. Extract standard top-level fields
    # ------------------------------------------------------------------
    status: str = str(visualization_payload.get("status", "READY"))
    raw_name = visualization_payload.get("dataset_name")
    dataset_name: str | None = str(raw_name) if raw_name is not None else None
    visualization_count: int = int(
        visualization_payload.get("visualization_count", 0)
    )
    raw_vizs = visualization_payload.get("visualizations") or []
    top_signals: list[str] = [
        str(s)
        for s in (visualization_payload.get("signals") or [])
        if s is not None
    ]

    # ------------------------------------------------------------------
    # 3. Pass through any extra Stage 8 fields (errors, clarification_options)
    # ------------------------------------------------------------------
    extra: dict[str, Any] = {}
    for key, value in visualization_payload.items():
        if key not in _STANDARD_PAYLOAD_KEYS:
            extra[key] = _to_json_safe(value)

    # ------------------------------------------------------------------
    # 4. Attach Stage 9 explanation to each visualization item
    # ------------------------------------------------------------------
    enriched: list[dict[str, Any]] = []
    if isinstance(raw_vizs, list):
        for item in raw_vizs:
            if isinstance(item, Mapping):
                enriched.append(_enrich_visualization_item(item, safe_signals))
            else:
                # Unexpected non-Mapping: convert and pass through
                enriched.append(_to_json_safe(item))

    # ------------------------------------------------------------------
    # 5. Assemble final response
    # ------------------------------------------------------------------
    response: dict[str, Any] = {
        "mode": mode,
        "status": status,
        "dataset_name": dataset_name,
        "visualization_count": visualization_count,
        "visualizations": enriched,
        "signals": top_signals,
    }
    # Merge extra Stage 8 fields (e.g. errors, clarification_options)
    response.update(extra)

    return _to_json_safe(response)