"""HTTP boundary for privacy-cleared dataset and report exports."""

from __future__ import annotations

import io
from typing import Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from api.export_service import ExportService


router = APIRouter(prefix="/api/runs/{run_id}/exports", tags=["exports"])


class ReportExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["pdf", "docx"]
    sections: list[Literal["overview", "profiling", "prediction", "insights", "visualization"]] = Field(min_length=1, max_length=5)
    prediction_id: str | None = Field(default=None, max_length=64)
    insights_conversation_id: str | None = Field(default=None, max_length=64)
    visualization_conversation_id: str | None = Field(default=None, max_length=64)


def _service(request: Request) -> ExportService:
    return request.app.state.export_service


def _headers(filename: str) -> dict[str, str]:
    return {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "no-store",
    }


@router.get("/availability")
def export_availability(
    run_id: str,
    request: Request,
    prediction_id: str | None = Query(default=None, max_length=64),
    insights_conversation_id: str | None = Query(default=None, max_length=64),
    visualization_conversation_id: str | None = Query(default=None, max_length=64),
):
    return _service(request).availability(
        run_id,
        prediction_id=prediction_id,
        insights_conversation_id=insights_conversation_id,
        visualization_conversation_id=visualization_conversation_id,
    )


@router.get("/dataset")
def export_dataset(
    run_id: str,
    request: Request,
    format: Literal["csv", "xlsx"] = Query(default="csv"),
):
    service = _service(request)
    if format == "csv":
        path, filename = service.dataset_csv(run_id)
        return FileResponse(
            path,
            media_type="text/csv; charset=utf-8",
            filename=filename,
            headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"},
        )
    stream, filename = service.dataset_xlsx(run_id)
    return StreamingResponse(
        stream,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=_headers(filename),
        background=BackgroundTask(stream.close),
    )


@router.post("/report")
def export_report(run_id: str, body: ReportExportRequest, request: Request):
    payload, filename, media_type = _service(request).report(
        run_id,
        format_name=body.format,
        sections=body.sections,
        prediction_id=body.prediction_id,
        insights_conversation_id=body.insights_conversation_id,
        visualization_conversation_id=body.visualization_conversation_id,
    )
    stream = io.BytesIO(payload)
    return StreamingResponse(
        stream,
        media_type=media_type,
        headers=_headers(filename),
        background=BackgroundTask(stream.close),
    )


__all__ = ["router"]
