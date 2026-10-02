"""Minimal FastAPI boundary for upload, run status, and Pipeline 1 results."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile

from intelligence.llm.provider import LLMProvider
from api.run_service import ApplicationRunService, ServiceError
from api.prediction_service import PredictionJobService
from api.routes.conversations import router as conversations_router
from api.routes.insights import router as insights_router
from api.routes.visualization import router as visualization_router
from api.schemas import (
    CreateRunResponse,
    FrontendPredictionResult,
    Pipeline1ResultResponse,
    PredictionCreateRequest,
    PredictionCreateResponse,
    PredictionRowsResponse,
    PredictionStatusResponse,
    RunStatusResponse,
)
from app.config import ApplicationSettings
from pipelines.pipeline3_results.conversation.orchestrator import (
    ConversationOrchestrator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"


def create_app(
    *,
    settings: ApplicationSettings | None = None,
    service: ApplicationRunService | None = None,
    prediction_service: PredictionJobService | None = None,
    semantic_llm_provider: LLMProvider | None = None,
) -> FastAPI:
    run_service = service or ApplicationRunService(settings)
    prediction_jobs = prediction_service or PredictionJobService(run_service)
    owns_prediction_jobs = prediction_service is None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        prediction_jobs.reconcile_startup()
        try:
            yield
        finally:
            if owns_prediction_jobs:
                # Do not hold application shutdown for long-running model work.
                # In-flight work has no checkpoint; a later startup reconciles it.
                prediction_jobs.shutdown(wait=False)
            if service is None:
                run_service.shutdown(wait=True)

    application = FastAPI(title="LYNX Application API", version="0.1.0", lifespan=lifespan)
    application.state.run_service = run_service
    application.state.prediction_service = prediction_jobs
    application.state.semantic_llm_provider = semantic_llm_provider
    application.state.conversation_orchestrator = ConversationOrchestrator(
        llm_provider=semantic_llm_provider
    )

    @application.exception_handler(ServiceError)
    async def controlled_error(_request: Request, exc: ServiceError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @application.exception_handler(RequestValidationError)
    async def controlled_validation_error(_request: Request, exc: RequestValidationError):
        insight_request_error = _request.url.path.startswith("/api/insights/")
        conversation_request_error = _request.url.path.startswith("/api/conversations/")
        pagination_error = any(
            tuple(error.get("loc", ()))[:1] == ("query",)
            and tuple(error.get("loc", ()))[-1:] in {("offset",), ("limit",)}
            for error in exc.errors()
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": (
                        "invalid_conversation_request"
                        if conversation_request_error
                        else "invalid_insight_request"
                        if insight_request_error
                        else "invalid_pagination"
                        if pagination_error
                        else "invalid_target"
                    ),
                    "message": (
                        "Conversation requests require valid JSON with a non-empty string field named 'message'."
                        if conversation_request_error
                        else "Insights requests require valid JSON with a non-empty string field named 'request_text'."
                        if insight_request_error
                        else "Prediction row pagination parameters are invalid."
                        if pagination_error
                        else "Target column must be a non-empty string."
                    ),
                }
            },
        )

    @application.post(
        "/api/runs", response_model=CreateRunResponse, status_code=202,
    )
    async def create_run(request: Request):
        try:
            form = await request.form()
        except Exception as exc:
            raise ServiceError(
                400, "invalid_multipart", "Request must contain one multipart dataset file."
            ) from exc
        files = [
            value for _key, value in form.multi_items()
            if isinstance(value, UploadFile)
        ]
        datasets = [
            value for key, value in form.multi_items()
            if key == "dataset" and isinstance(value, UploadFile)
        ]
        if len(datasets) != 1 or len(files) != 1:
            for uploaded in files:
                await uploaded.close()
            raise ServiceError(
                400, "invalid_dataset_upload_count",
                "Exactly one file named 'dataset' must be uploaded.",
            )
        return await run_service.create_from_upload(datasets[0])

    @application.get("/api/runs/{run_id}", response_model=RunStatusResponse)
    def run_status(run_id: str):
        return run_service.status(run_id)

    @application.get(
        "/api/runs/{run_id}/pipeline1", response_model=Pipeline1ResultResponse,
    )
    def pipeline1_result(run_id: str):
        return run_service.pipeline1_result(run_id)

    @application.post(
        "/api/runs/{run_id}/predictions",
        response_model=PredictionCreateResponse,
        status_code=202,
    )
    def create_prediction(run_id: str, request: PredictionCreateRequest):
        return prediction_jobs.create(run_id, request.target_column)

    @application.get(
        "/api/runs/{run_id}/predictions/{prediction_id}",
        response_model=PredictionStatusResponse,
    )
    def prediction_status(run_id: str, prediction_id: str):
        return prediction_jobs.status(run_id, prediction_id)

    @application.get(
        "/api/runs/{run_id}/predictions/{prediction_id}/result",
        response_model=FrontendPredictionResult,
    )
    def prediction_result(run_id: str, prediction_id: str):
        return prediction_jobs.result(run_id, prediction_id)

    @application.get(
        "/api/runs/{run_id}/predictions/{prediction_id}/rows",
        response_model=PredictionRowsResponse,
    )
    def prediction_rows(
        run_id: str,
        prediction_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ):
        return prediction_jobs.rows(
            run_id, prediction_id, offset=offset, limit=limit
        )

    @application.get("/api/runs/{run_id}/predictions/{prediction_id}/export")
    def prediction_export(run_id: str, prediction_id: str):
        chunks = prediction_jobs.export(run_id, prediction_id)
        safe_id = prediction_jobs.store.validate_prediction_id(prediction_id)
        return StreamingResponse(
            chunks,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="lynx_prediction_{safe_id}.csv"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    # Visualization routes -- Pipeline 3 Integration
    application.include_router(visualization_router)
    application.include_router(insights_router)
    application.include_router(conversations_router)

    # Keep the browser and API on one origin. API routes are registered first
    # so this catch-all static mount cannot shadow them.
    application.mount("/", StaticFiles(directory=FRONTEND_ROOT, html=True), name="frontend")

    return application


app = create_app()


__all__ = ["app", "create_app"]
