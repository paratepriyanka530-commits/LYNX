"""Frozen contracts for deterministic Insights landing intelligence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan


SummaryStatus = Literal["READY", "PARTIAL", "EMPTY"]
LandingStatus = Literal["READY", "PARTIAL", "EMPTY"]
SummaryFactType = Literal[
    "DATASET_OVERVIEW", "DATA_QUALITY", "SEMANTIC_STRUCTURE",
    "MEASURE_AGGREGATE", "GROUP_RANKING", "ENTITY_COUNT",
    "NUMERIC_RANGE", "TREND_OPPORTUNITY", "RELATIONSHIP_OPPORTUNITY",
    "CATEGORY_OPPORTUNITY",
]


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", str_strip_whitespace=True
    )


class SummaryFact(StrictFrozenModel):
    fact_id: str = Field(pattern=r"^fact_[0-9a-f]{16}$")
    fact_type: SummaryFactType
    importance: float = Field(ge=0.0, le=1.0)
    values: dict[str, JsonValue] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class SummaryPackage(StrictFrozenModel):
    dataset_id: str | None = None
    status: SummaryStatus
    facts: list[SummaryFact] = Field(default_factory=list, max_length=5)
    opportunities: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class NaturalLanguageSummary(StrictFrozenModel):
    text: str = Field(min_length=1)
    language: str = Field(min_length=2, max_length=35)
    status: Literal["READY", "FALLBACK"]
    fallback_used: bool
    caveats: list[str] = Field(default_factory=list)


class SummaryModelDraft(StrictFrozenModel):
    text: str = Field(min_length=1, max_length=3000)
    language: str = Field(min_length=2, max_length=35)


class SuggestedQuestion(StrictFrozenModel):
    id: str = Field(pattern=r"^question_[0-9a-f]{16}$")
    text: str = Field(min_length=1)
    analytical_goal: str = Field(min_length=1)
    plan_template: AnalyticalPlan
    answerable: bool
    rationale: str = Field(min_length=1)


class SuggestedQuestionItem(StrictFrozenModel):
    id: str
    text: str
    analytical_goal: str


class InsightsLandingState(StrictFrozenModel):
    dataset: str
    status: LandingStatus
    summary_text: str
    summary: SummaryPackage
    suggested_questions: list[SuggestedQuestionItem] = Field(
        default_factory=list, max_length=5
    )
    signals: list[str] = Field(default_factory=list)


__all__ = [
    "InsightsLandingState", "LandingStatus", "NaturalLanguageSummary",
    "SuggestedQuestion", "SuggestedQuestionItem", "SummaryFact",
    "SummaryFactType", "SummaryModelDraft", "SummaryPackage", "SummaryStatus",
]
