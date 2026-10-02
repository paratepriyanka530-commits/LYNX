"""Thin HTTP boundary for deterministic Pipeline 3 Insights."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from api.dataset_resolver import resolve_cleaned_dataset
from api.run_service import ServiceError
from api.schemas import InsightQueryRequest
from pipelines.pipeline3_results.insights.results_service import (
    build_insight_results,
)
from pipelines.pipeline3_results.insights.schemas import InsightResponse


router = APIRouter(prefix="/api/insights", tags=["insights"])


def _get_runtime_root(request: Request) -> Path | None:
    """Use the application run service's configured artifact root."""
    run_service = getattr(request.app.state, "run_service", None)
    settings = getattr(run_service, "settings", None)
    runtime_root = getattr(settings, "runtime_root", None)
    return Path(runtime_root) if runtime_root is not None else None


@router.post("/{dataset_id}/query", response_model=InsightResponse)
def query_insights(
    dataset_id: str,
    body: InsightQueryRequest,
    request: Request,
) -> dict[str, object]:
    """Answer one factual query against an already-cleaned dataset."""
    try:
        resolved = resolve_cleaned_dataset(
            dataset_id=dataset_id,
            runtime_root=_get_runtime_root(request),
        )
        return build_insight_results(
            dataframe=resolved.dataframe,
            question=body.request_text,
            pipeline1_report=resolved.pipeline1_report,
            dataset_name=resolved.dataset_name,
            llm_provider=getattr(request.app.state, "semantic_llm_provider", None),
        )
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError(
            500,
            "insight_generation_error",
            "Failed to generate the insight response.",
        ) from exc


__all__ = ["router"]
