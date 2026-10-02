"""API boundary for deterministic structured Pipeline 3 conversations."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from api.dataset_resolver import resolve_cleaned_dataset
from api.run_service import ServiceError
from pipelines.pipeline3_results.conversation.context_store import (
    ConversationDatasetMismatchError,
    ConversationNotFoundError,
)
from pipelines.pipeline3_results.conversation.orchestrator import (
    ConversationOrchestrator,
)
from pipelines.pipeline3_results.conversation.schemas import (
    ConversationRequest,
    ConversationResult,
)


router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _runtime_root(request: Request) -> Path | None:
    service = getattr(request.app.state, "run_service", None)
    settings = getattr(service, "settings", None)
    root = getattr(settings, "runtime_root", None)
    return Path(root) if root is not None else None


def _orchestrator(request: Request) -> ConversationOrchestrator:
    return request.app.state.conversation_orchestrator


def _resolved(dataset_id: str, request: Request):
    return resolve_cleaned_dataset(
        dataset_id=dataset_id, runtime_root=_runtime_root(request)
    )


@router.post("/{dataset_id}", response_model=ConversationResult)
def create_conversation(
    dataset_id: str, body: ConversationRequest, request: Request
) -> ConversationResult:
    try:
        dataset = _resolved(dataset_id, request)
        return _orchestrator(request).start(
            dataset_id,
            dataset.dataframe,
            body.message,
            pipeline1_report=dataset.pipeline1_report,
            dataset_name=dataset.dataset_name,
        )
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError(
            500, "conversation_error", "Failed to process the conversation request."
        ) from exc


@router.post("/{dataset_id}/{conversation_id}", response_model=ConversationResult)
def continue_conversation(
    dataset_id: str,
    conversation_id: str,
    body: ConversationRequest,
    request: Request,
) -> ConversationResult:
    try:
        # Check ownership before loading a potentially unrelated dataset.
        _orchestrator(request).store.get(dataset_id, conversation_id)
        dataset = _resolved(dataset_id, request)
        return _orchestrator(request).continue_conversation(
            dataset_id,
            conversation_id,
            dataset.dataframe,
            body.message,
            pipeline1_report=dataset.pipeline1_report,
            dataset_name=dataset.dataset_name,
        )
    except ConversationNotFoundError as exc:
        raise ServiceError(
            404, "conversation_not_found", "The conversation does not exist."
        ) from exc
    except ConversationDatasetMismatchError as exc:
        raise ServiceError(
            409,
            "conversation_dataset_mismatch",
            "The conversation belongs to a different dataset.",
        ) from exc
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError(
            500, "conversation_error", "Failed to process the conversation request."
        ) from exc


__all__ = ["router"]
