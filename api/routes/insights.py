"""Thin HTTP boundary for deterministic Pipeline 3 Insights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, Query, Request

from api.dataset_resolver import resolve_cleaned_dataset
from api.run_service import ServiceError
from api.schemas import InsightQueryRequest
from pipelines.pipeline3_results.insights.results_service import (
    build_insight_results,
)
from pipelines.pipeline3_results.insights.schemas import InsightResponse
from pipelines.pipeline3_results.insights.landing.schemas import InsightsLandingState
from pipelines.pipeline3_results.insights.landing.service import InsightsLandingService


router = APIRouter(prefix="/api/insights", tags=["insights"])


def _get_runtime_root(request: Request) -> Path | None:
    """Use the application run service's configured artifact root."""
    run_service = getattr(request.app.state, "run_service", None)
    settings = getattr(run_service, "settings", None)
    runtime_root = getattr(settings, "runtime_root", None)
    return Path(runtime_root) if runtime_root is not None else None


@router.get("/{dataset_id}/landing", response_model=InsightsLandingState)
def insights_landing(
    dataset_id: str,
    request: Request,
    language: str = Query(default="en", min_length=2, max_length=35),
) -> InsightsLandingState:
    """Return cached verified summary facts and answerable questions."""
    try:
        resolved = resolve_cleaned_dataset(
            dataset_id=dataset_id,
            runtime_root=_get_runtime_root(request),
        )
        stat = resolved.csv_path.stat()
        report_digest = hashlib.sha256(json.dumps(
            resolved.pipeline1_report or {},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()[:16]
        data_version = f"{stat.st_size}:{stat.st_mtime_ns}:{report_digest}"
        service = getattr(request.app.state, "insights_landing_service", None)
        if service is None:
            service = InsightsLandingService(
                getattr(request.app.state, "semantic_llm_provider", None)
            )
        payload = service.build(
            dataset_id,
            resolved.dataframe,
            pipeline1_report=resolved.pipeline1_report,
            dataset_name=resolved.dataset_name,
            requested_language=language,
            data_version=data_version,
        )
        snapshots = getattr(request.app.state, "export_artifacts", None)
        if snapshots is not None:
            snapshots.set_landing(dataset_id, payload.model_dump(mode="json"))
        return payload
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError(
            500,
            "insights_landing_error",
            "Failed to generate the Insights landing state.",
        ) from exc


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
            response_provider=getattr(request.app.state, "semantic_llm_provider", None),
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
