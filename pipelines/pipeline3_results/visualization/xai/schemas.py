"""Frozen public contracts for chart explanations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


FactType = Literal[
    "MAXIMUM", "MINIMUM", "RANKING", "TIE", "ABSOLUTE_CHANGE",
    "PERCENT_CHANGE", "TOP_SHARE", "CONCENTRATION", "TREND_DIRECTION",
    "LARGEST_INCREASE", "LARGEST_DECREASE", "CORRELATION", "PAIR_COUNT",
    "DOMINANT_BIN", "RANGE", "MEDIAN", "IQR", "OUTLIER_COUNT",
    "HIGH_VALUE_CELL", "LOW_VALUE_CELL", "COVERAGE", "LIMITATION",
]


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class XAIFact(StrictFrozenModel):
    fact_id: str = Field(pattern=r"^xai_fact_[0-9a-f]{16}$")
    fact_type: FactType
    importance: float = Field(ge=0.0, le=1.0)
    values: dict[str, JsonValue]
    evidence_refs: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class XAIResult(StrictFrozenModel):
    visualization_id: str
    status: Literal["READY", "PARTIAL", "UNSUPPORTED", "EMPTY"]
    why_selected: str | None = None
    why_this_chart: str
    what_it_shows: str
    facts: list[XAIFact] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    language: str = Field(min_length=2, max_length=35)
    fallback_used: bool = False


class XAIModelDraft(StrictFrozenModel):
    why_this_chart: str = Field(min_length=1, max_length=500)
    what_it_shows: str = Field(min_length=1, max_length=1200)
    limitations: list[str] = Field(default_factory=list, max_length=5)
    language: str = Field(min_length=2, max_length=35)

    @field_validator("why_this_chart", "what_it_shows")
    @classmethod
    def plain_text_only(cls, value: str) -> str:
        if "```" in value:
            raise ValueError("code blocks are not valid XAI wording")
        return value


__all__ = ["FactType", "XAIFact", "XAIModelDraft", "XAIResult"]
