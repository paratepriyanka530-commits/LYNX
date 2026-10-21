"""Frozen contracts for verified-result response generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


ResponseGenerationStatus = Literal[
    "READY", "PARTIAL", "FALLBACK", "GENERATION_FAILED"
]


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", str_strip_whitespace=True
    )


class NaturalLanguageResponse(StrictFrozenModel):
    """Small provider-independent answer contract."""

    text: str = Field(min_length=1)
    language: str = Field(min_length=2, max_length=35)
    status: ResponseGenerationStatus
    verified_result_id: str
    caveats: list[str] = Field(default_factory=list)
    fallback_used: bool = False


class ResponseContext(StrictFrozenModel):
    """Minimal deterministic facts that may be disclosed to the response model."""

    original_query: str = Field(min_length=1)
    semantic_intent_summary: dict[str, JsonValue] = Field(default_factory=dict)
    verified_result: dict[str, JsonValue]
    semantic_labels: dict[str, str] = Field(default_factory=dict)
    units_or_formats_if_known: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    conversation_context: dict[str, JsonValue] = Field(default_factory=dict)
    requested_language: str = Field(min_length=2, max_length=35)
    truncated: bool = False


class ModelResponseDraft(StrictFrozenModel):
    """Strict structured output requested from the existing LLMProvider."""

    text: str = Field(min_length=1, max_length=2000)
    language: str = Field(min_length=2, max_length=35)

    @field_validator("text")
    @classmethod
    def text_must_be_plain(cls, value: str) -> str:
        if "```" in value:
            raise ValueError("code blocks are not valid answer text")
        return value


class ResponseValidation(StrictFrozenModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


__all__ = [
    "ModelResponseDraft",
    "NaturalLanguageResponse",
    "ResponseContext",
    "ResponseGenerationStatus",
    "ResponseValidation",
]
