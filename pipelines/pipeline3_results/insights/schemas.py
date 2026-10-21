"""Frozen, JSON-safe contracts for deterministic Pipeline 3 Insights."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from pipelines.pipeline3_results.nlp.schemas import (
    Aggregation,
    ComparisonOperation,
    QueryFilter,
    FailureCode,
)


InsightStatus = Literal["READY", "NEEDS_CLARIFICATION", "INVALID"]
InsightValidationStatus = Literal["VALID", "NEEDS_CLARIFICATION", "INVALID"]
InsightQueryType = Literal[
    "SIMPLE_AGGREGATION",
    "GROUPED_AGGREGATION",
    "GROUPED_COMPARISON",
    "TOP_N",
    "BOTTOM_N",
]
CountMode = Literal["COUNT_ROWS", "COUNT_NON_NULL", "COUNT_DISTINCT"]


class InsightQuery(BaseModel):
    """Executable deterministic query derived from a shared interpretation."""

    model_config = ConfigDict(frozen=True)

    original_text: str
    query_type: InsightQueryType
    measure_column: str | None = None
    dimension_column: str | None = None
    aggregation: Aggregation
    comparison: ComparisonOperation = "NONE"
    limit: int | None = Field(default=None, ge=1)
    count_mode: CountMode | None = None
    filters: list[QueryFilter] = Field(default_factory=list)
    temporal_columns: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)


class InsightAdaptation(BaseModel):
    """Internal result of adapting a canonical NLP interpretation."""

    model_config = ConfigDict(frozen=True)

    status: InsightStatus
    failure_code: FailureCode | None = None
    query: InsightQuery | None = None
    message: str
    clarification_options: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)


class InsightValidation(BaseModel):
    """Validation outcome before any dataframe computation is performed."""

    model_config = ConfigDict(frozen=True)

    status: InsightValidationStatus
    failure_code: FailureCode | None = None
    query: InsightQuery | None = None
    message: str
    errors: list[str] = Field(default_factory=list)
    clarification_options: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)


class InsightRow(BaseModel):
    """One factual grouped result row."""

    model_config = ConfigDict(frozen=True)

    group: JsonValue
    value: float | int


class InsightEvidence(BaseModel):
    """Coverage and grouped values used to produce an answer."""

    model_config = ConfigDict(frozen=True)

    total_rows: int
    usable_rows: int
    excluded_null_rows: int
    group_count: int | None = None
    grouped_values: list[InsightRow] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)


class InsightResult(BaseModel):
    """Pure computed result returned by the deterministic analysis engine."""

    model_config = ConfigDict(frozen=True)

    query_type: InsightQueryType
    measure_column: str | None = None
    dimension_column: str | None = None
    aggregation: Aggregation
    comparison: ComparisonOperation
    count_mode: CountMode | None = None
    value: float | int | None = None
    winner: JsonValue | None = None
    rows: list[InsightRow] = Field(default_factory=list)
    evidence: InsightEvidence
    has_data: bool = True


class InsightResponse(BaseModel):
    """Frontend-safe deterministic answer contract."""

    model_config = ConfigDict(frozen=True)

    status: InsightStatus
    failure_code: FailureCode | None = None
    answer: str | None = None
    result: InsightResult | None = None
    evidence: InsightEvidence | None = None
    message: str | None = None
    clarification_options: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)


__all__ = [
    "CountMode",
    "InsightAdaptation",
    "InsightEvidence",
    "InsightQuery",
    "InsightQueryType",
    "InsightResponse",
    "InsightResult",
    "InsightRow",
    "InsightStatus",
    "InsightValidation",
    "InsightValidationStatus",
]
