"""Small canonical contracts for pretrained-first semantic interpretation."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pipelines.pipeline3_results.nlp.chart_types import RequestedChart


SemanticIntentType = Literal["INSIGHT", "VISUALIZATION", "UNKNOWN"]
AnalyticalGoal = Literal[
    "SIMPLE_AGGREGATION",
    "GROUP_COMPARISON",
    "RANK_GROUPS",
    "TOP_N",
    "BOTTOM_N",
    "TREND",
    "DISTRIBUTION",
    "RELATIONSHIP",
    "COUNT_ENTITIES",
    "PART_TO_WHOLE",
    "EXPLORATORY",
    "UNKNOWN",
]
SemanticAggregation = Literal["SUM", "MEAN", "COUNT", "MIN", "MAX", "MEDIAN", "NONE"]
SemanticComparison = Literal["HIGHEST", "LOWEST", "TOP_N", "BOTTOM_N", "NONE"]
SemanticFilterOperator = Literal[
    "EQUALS",
    "NOT_EQUALS",
    "GREATER_THAN",
    "GREATER_THAN_OR_EQUAL",
    "LESS_THAN",
    "LESS_THAN_OR_EQUAL",
    "CONTAINS",
    "BEFORE",
    "AFTER",
    "BETWEEN",
    "UNKNOWN",
]
SemanticSlot = Literal[
    "measure", "dimension", "aggregation", "comparison", "time_expression"
]
InterpretationState = Literal[
    "VALID_INTENT",
    "INVALID_MODEL_OUTPUT",
    "PROVIDER_UNAVAILABLE",
    "PROVIDER_TIMEOUT",
    "PROVIDER_ERROR",
]
InterpretationSource = Literal["LLM", "DETERMINISTIC_FALLBACK"]


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class SemanticFilter(StrictFrozenModel):
    field_concept: str = Field(min_length=1)
    operator: SemanticFilterOperator
    value_concept: str | None = None


class MissingSlot(StrictFrozenModel):
    slot: SemanticSlot
    reason: str = Field(min_length=1, max_length=300)


class SemanticAmbiguity(StrictFrozenModel):
    slot: SemanticSlot
    concept: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=300)


class SemanticIntent(StrictFrozenModel):
    """User meaning before dataset grounding, planning, or execution."""

    intent: SemanticIntentType
    analytical_goal: AnalyticalGoal
    measure_concepts: list[str] = Field(default_factory=list)
    dimension_concepts: list[str] = Field(default_factory=list)
    aggregation: SemanticAggregation = "NONE"
    comparison: SemanticComparison = "NONE"
    limit: int | None = Field(default=None, ge=1)
    requested_chart: RequestedChart | None = None
    filters: list[SemanticFilter] = Field(default_factory=list)
    time_expression: str | None = None
    missing_slots: list[MissingSlot] = Field(default_factory=list)
    ambiguities: list[SemanticAmbiguity] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None
    unsupported: bool = False
    unsupported_reason: str | None = None

    @field_validator("measure_concepts", "dimension_concepts")
    @classmethod
    def normalize_concepts(cls, concepts: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for concept in concepts:
            clean = concept.strip()
            if not clean:
                raise ValueError("semantic concepts must not be empty")
            if re.search(
                r"(?:^|\W)(?:import\s+\w+|exec\s*\(|eval\s*\(|__\w+__)",
                clean,
                re.IGNORECASE,
            ):
                raise ValueError("semantic concepts must not contain executable code")
            key = clean.casefold()
            if key not in seen:
                seen.add(key)
                normalized.append(clean)
        return normalized

    @model_validator(mode="after")
    def validate_contract(self) -> "SemanticIntent":
        missing = {item.slot for item in self.missing_slots}
        ambiguous = {item.slot for item in self.ambiguities}
        overlap = missing & ambiguous
        if overlap:
            raise ValueError(
                f"slots cannot be both missing and ambiguous: {sorted(overlap)}"
            )
        if "measure" in missing and self.measure_concepts:
            raise ValueError("a missing measure slot cannot contain measure concepts")
        if "dimension" in missing and self.dimension_concepts:
            raise ValueError("a missing dimension slot cannot contain dimension concepts")
        if self.needs_clarification:
            if not self.clarification_question:
                raise ValueError("clarification_question is required")
            if not self.missing_slots and not self.ambiguities:
                raise ValueError("clarification requires a missing or ambiguous slot")
        else:
            if self.clarification_question is not None:
                raise ValueError(
                    "clarification_question must be null when clarification is false"
                )
            if self.missing_slots or self.ambiguities:
                raise ValueError(
                    "missing or ambiguous slots require clarification"
                )
        if self.unsupported:
            if not self.unsupported_reason:
                raise ValueError("unsupported_reason is required")
            if self.needs_clarification:
                raise ValueError("unsupported requests cannot also request clarification")
        elif self.unsupported_reason is not None:
            raise ValueError("unsupported_reason must be null when supported")
        if self.limit is not None and self.comparison not in {"TOP_N", "BOTTOM_N"}:
            raise ValueError("a numeric limit requires TOP_N or BOTTOM_N comparison")
        return self


class SemanticInterpretationResult(StrictFrozenModel):
    state: InterpretationState
    source: InterpretationSource
    semantic_intent: SemanticIntent | None = None
    message: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "SemanticInterpretationResult":
        if self.state == "VALID_INTENT" and self.semantic_intent is None:
            raise ValueError("VALID_INTENT requires semantic_intent")
        if self.state != "VALID_INTENT" and self.semantic_intent is not None:
            raise ValueError("provider failures cannot contain semantic_intent")
        return self


__all__ = [
    "AnalyticalGoal",
    "InterpretationSource",
    "InterpretationState",
    "MissingSlot",
    "RequestedChart",
    "SemanticAggregation",
    "SemanticAmbiguity",
    "SemanticComparison",
    "SemanticFilter",
    "SemanticFilterOperator",
    "SemanticIntent",
    "SemanticIntentType",
    "SemanticInterpretationResult",
    "SemanticSlot",
]
