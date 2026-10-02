"""Strict, non-executable schemas for optional language interpretation."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pipelines.pipeline3_results.nlp.schemas import (
    Aggregation,
    ComparisonOperation,
    FilterOperator,
    QueryIntent,
    SharedChartType,
)


_EXECUTABLE = re.compile(
    r"(?:\b(?:import|exec|eval|lambda|select|insert|update|delete|drop)\b|"
    r"__|\bpd\.|\bdf\s*\[|```|;)",
    re.IGNORECASE,
)


def _safe_language(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("semantic terms must be non-empty")
    if len(cleaned) > 160:
        raise ValueError("semantic terms must be at most 160 characters")
    if _EXECUTABLE.search(cleaned):
        raise ValueError("executable content is forbidden")
    return cleaned


class LLMQueryFilter(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    column_term: str
    operator: FilterOperator
    value_term: str

    _validate_column = field_validator("column_term")(_safe_language)
    _validate_value = field_validator("value_term")(_safe_language)


class LLMQueryInterpretation(BaseModel):
    """Language concepts only; fields can never contain executable behavior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    supported: bool
    unsupported_reason: str | None = None
    intent: QueryIntent
    measure_terms: list[str] = Field(default_factory=list, max_length=4)
    dimension_terms: list[str] = Field(default_factory=list, max_length=4)
    aggregation: Aggregation = "NONE"
    comparison: ComparisonOperation = "NONE"
    requested_chart: SharedChartType | None = None
    filters: list[LLMQueryFilter] = Field(default_factory=list, max_length=4)
    time_terms: list[str] = Field(default_factory=list, max_length=4)
    limit: int | None = Field(default=None, ge=1, le=100)

    @field_validator("measure_terms", "dimension_terms", "time_terms")
    @classmethod
    def validate_terms(cls, values: list[str]) -> list[str]:
        return [_safe_language(value) for value in values]

    @field_validator("unsupported_reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        return _safe_language(value) if value is not None else None

    @model_validator(mode="after")
    def validate_supported_shape(self):
        if self.supported and self.intent == "UNKNOWN":
            raise ValueError("supported interpretations require a concrete intent")
        if not self.supported and not self.unsupported_reason:
            raise ValueError("unsupported interpretations require a reason")
        if self.supported and not (
            self.measure_terms or self.dimension_terms or self.time_terms
        ):
            raise ValueError("supported interpretations require analytical entities")
        if self.comparison in {"HIGHEST", "LOWEST", "TOP_N", "BOTTOM_N"} and not (
            self.measure_terms and self.dimension_terms
        ):
            raise ValueError("comparisons require both a measure and a dimension")
        if self.comparison in {"TOP_N", "BOTTOM_N"} and self.limit is None:
            raise ValueError("TOP_N and BOTTOM_N require a limit")
        if self.limit is not None and self.comparison not in {"TOP_N", "BOTTOM_N"}:
            raise ValueError("limit is only supported for TOP_N or BOTTOM_N")
        return self


FollowupModification = Literal[
    "REPLACE_MEASURE", "REPLACE_DIMENSION", "SET_AGGREGATION",
    "SET_COMPARISON", "SET_CHART", "SET_LIMIT", "ADD_FILTER",
    "SET_TEMPORAL", "UNSUPPORTED",
]


class LLMFollowupInterpretation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    supported: bool
    modification: FollowupModification
    replace_measure_term: str | None = None
    replace_dimension_term: str | None = None
    aggregation: Aggregation | None = None
    comparison: ComparisonOperation | None = None
    requested_chart: SharedChartType | None = None
    limit: int | None = Field(default=None, ge=1, le=100)
    filter: LLMQueryFilter | None = None
    temporal_terms: list[str] = Field(default_factory=list, max_length=4)
    unsupported_reason: str | None = None

    @field_validator(
        "replace_measure_term", "replace_dimension_term", "unsupported_reason"
    )
    @classmethod
    def validate_optional_term(cls, value: str | None) -> str | None:
        return _safe_language(value) if value is not None else None

    @field_validator("temporal_terms")
    @classmethod
    def validate_time_terms(cls, values: list[str]) -> list[str]:
        return [_safe_language(value) for value in values]

    @model_validator(mode="after")
    def validate_modification_payload(self):
        required = {
            "REPLACE_MEASURE": self.replace_measure_term,
            "REPLACE_DIMENSION": self.replace_dimension_term,
            "SET_AGGREGATION": self.aggregation,
            "SET_COMPARISON": self.comparison,
            "SET_CHART": self.requested_chart,
            "SET_LIMIT": self.limit,
            "ADD_FILTER": self.filter,
            "SET_TEMPORAL": self.temporal_terms or None,
            "UNSUPPORTED": self.unsupported_reason,
        }
        if required[self.modification] is None:
            raise ValueError("the selected modification is missing its required value")
        if not self.supported and self.modification != "UNSUPPORTED":
            raise ValueError("unsupported follow-ups must use UNSUPPORTED")
        return self


__all__ = [
    "FollowupModification",
    "LLMFollowupInterpretation",
    "LLMQueryFilter",
    "LLMQueryInterpretation",
]
