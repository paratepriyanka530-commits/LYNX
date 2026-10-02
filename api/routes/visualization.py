"""Visualization API routes -- Pipeline 3 Integration.

Provides two clean, synchronous, deterministic HTTP endpoints for already-cleaned datasets:
1. GET /api/visualizations/{dataset_id} (Automatic Mode)
2. POST /api/visualizations/{dataset_id}/query (Requested Mode)

Rules:
- Strictly operates on already-cleaned datasets (never triggers Pipeline 1).
- Returns the EXACT JSON payload shape produced by build_visualization_results.
- Legitimate outcomes (READY, NO_MEANINGFUL_VISUALIZATION, NEEDS_CLARIFICATION, INVALID)
  return HTTP 200 with the exact status and payload.
- Returns HTTP 404 if dataset_id cannot be resolved to a known cleaned dataset on disk.
- Returns HTTP 400 for malformed request bodies.
- Returns HTTP 500 with a clear message only for unexpected internal errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from api.dataset_resolver import resolve_cleaned_dataset
from api.run_service import ServiceError
from pipelines.pipeline3_results.visualization.results_service import (
    build_visualization_results,
)

router = APIRouter(prefix="/api/visualizations", tags=["visualizations"])


def _get_runtime_root(request: Request) -> Path | None:
    """Extract configured runtime root from app state if available."""
    try:
        run_service = getattr(request.app.state, "run_service", None)
        if run_service and hasattr(run_service, "settings"):
            return Path(run_service.settings.runtime_root)
    except Exception:
        pass
    return None


@router.api_route("/{dataset_id}", methods=["GET", "POST"])
async def get_automatic_visualizations(
    dataset_id: str,
    request: Request,
    max_visualizations: int = Query(default=5, ge=1, le=50),
) -> JSONResponse:
    """Generate automatic visualizations for an already-cleaned dataset (Mode A).

    Loads the cleaned dataset from its known location on disk and invokes
    Stage 10 build_visualization_results in automatic mode (request_text=None).
    """
    runtime_root = _get_runtime_root(request)
    resolved = resolve_cleaned_dataset(
        dataset_id=dataset_id,
        runtime_root=runtime_root,
    )

    try:
        payload = build_visualization_results(
            dataframe=resolved.dataframe,
            request_text=None,
            pipeline1_report=resolved.pipeline1_report,
            dataset_name=resolved.dataset_name,
            max_visualizations=max_visualizations,
            llm_provider=getattr(request.app.state, "semantic_llm_provider", None),
        )
    except Exception as exc:
        raise ServiceError(
            500,
            "visualization_generation_error",
            f"Failed to generate automatic visualizations: {exc}",
        ) from exc

    return JSONResponse(status_code=200, content=payload)


@router.post("/{dataset_id}/query")
async def query_visualization(
    dataset_id: str,
    request: Request,
) -> JSONResponse:
    """Resolve a natural-language visualization query on an already-cleaned dataset (Mode B).

    Accepts a JSON body: {"request_text": "..."}
    Invokes Stage 10 build_visualization_results with the supplied query.
    """
    # -----------------------------------------------------------------------
    # 1. Parse and validate JSON request body
    # -----------------------------------------------------------------------
    try:
        raw_body = await request.body()
        if not raw_body:
            raise ServiceError(
                400,
                "malformed_request",
                "Request body is required and must be valid JSON containing 'request_text'.",
            )
        body = json.loads(raw_body.decode("utf-8"))
    except ServiceError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ServiceError(
            400,
            "malformed_request",
            f"Request body must be valid JSON: {exc}",
        ) from exc
    except Exception as exc:
        raise ServiceError(
            400,
            "malformed_request",
            f"Could not read request body: {exc}",
        ) from exc

    if not isinstance(body, dict):
        raise ServiceError(
            400,
            "malformed_request",
            "Request body must be a JSON object with a 'request_text' field.",
        )

    if "request_text" not in body:
        raise ServiceError(
            400,
            "missing_request_text",
            "Missing required field 'request_text' in request body.",
        )

    request_text = body.get("request_text")
    if not isinstance(request_text, str) or not request_text.strip():
        raise ServiceError(
            400,
            "invalid_request_text",
            "'request_text' must be a non-empty string.",
        )

    # -----------------------------------------------------------------------
    # 2. Resolve dataset and execute Stage 10
    # -----------------------------------------------------------------------
    runtime_root = _get_runtime_root(request)
    resolved = resolve_cleaned_dataset(
        dataset_id=dataset_id,
        runtime_root=runtime_root,
    )

    try:
        payload = build_visualization_results(
            dataframe=resolved.dataframe,
            request_text=request_text.strip(),
            pipeline1_report=resolved.pipeline1_report,
            dataset_name=resolved.dataset_name,
            llm_provider=getattr(request.app.state, "semantic_llm_provider", None),
        )
    except Exception as exc:
        raise ServiceError(
            500,
            "visualization_generation_error",
            f"Failed to resolve visualization query: {exc}",
        ) from exc

    return JSONResponse(status_code=200, content=payload)
