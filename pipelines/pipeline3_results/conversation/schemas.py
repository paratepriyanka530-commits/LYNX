"""Frozen contracts for deterministic multi-turn Pipeline 3 queries."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pipelines.pipeline3_results.nlp.schemas import (
    Aggregation,
    ComparisonOperation,
    QueryFilter,
    QueryIntent,
    SharedChartType,
    SharedChartVariant,
)


ConversationStatus = Literal[
    "READY", "NEEDS_CLARIFICATION", "INVALID", "NO_MEANINGFUL_VISUALIZATION"
]
ClarificationSlot = Literal["measure", "dimension", "time", "filter"]


class ConversationQueryState(BaseModel):
    """Minimal structured request state retained between turns."""

    model_config = ConfigDict(frozen=True)

    intent: QueryIntent
    measure: str | None = None
    dimension: str | None = None
    measure_term: str | None = None
    dimension_term: str | None = None
    aggregation: Aggregation = "NONE"
    comparison: ComparisonOperation = "NONE"
    limit: int | None = Field(default=None, ge=1)
    filters: tuple[QueryFilter, ...] = ()
    temporal_terms: tuple[str, ...] = ()
    requested_chart: SharedChartType | None = None
    requested_chart_variant: SharedChartVariant | None = None


class PendingClarification(BaseModel):
    model_config = ConfigDict(frozen=True)

    term: str
    slot: ClarificationSlot
    choices: tuple[str, ...]
    originating_query: ConversationQueryState


class ConversationTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    turn_id: str
    user_text: str
    intent: QueryIntent
    status: ConversationStatus
    query: ConversationQueryState | None = None
    result_metadata: dict[str, Any] = Field(default_factory=dict)
    signals: tuple[str, ...] = ()


class ConversationContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: str
    conversation_id: str
    active_query: ConversationQueryState | None = None
    pending_clarification: PendingClarification | None = None
    turns: tuple[ConversationTurn, ...] = ()


class ConversationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str

    @field_validator("message")
    @classmethod
    def message_must_be_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must be a non-empty string")
        return stripped


class ConversationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: str
    turn_id: str
    intent: QueryIntent
    status: ConversationStatus
    message: str | None = None
    visualization: dict[str, Any] | None = None
    insight: dict[str, Any] | None = None
    clarification_options: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)


class FollowupResolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    is_followup: bool
    modification: str | None = None
    query: ConversationQueryState | None = None
    signals: tuple[str, ...] = ()


__all__ = [
    "ClarificationSlot",
    "ConversationContext",
    "ConversationQueryState",
    "ConversationRequest",
    "ConversationResult",
    "ConversationStatus",
    "ConversationTurn",
    "FollowupResolution",
    "PendingClarification",
]
