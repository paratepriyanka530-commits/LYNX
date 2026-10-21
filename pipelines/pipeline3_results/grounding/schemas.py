"""Canonical contracts for dataset knowledge and semantic grounding."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


GroundingStatus = Literal["RESOLVED", "AMBIGUOUS", "UNRESOLVED", "UNSUPPORTED"]
GroundingSlot = Literal["MEASURE", "DIMENSION", "FILTER", "TIME"]
ExpressionKind = Literal[
    "COLUMN",
    "AGGREGATION",
    "COUNT_ROWS",
    "COUNT_NON_NULL",
    "COUNT_DISTINCT",
    "TIME_BUCKET",
    "STATISTIC",
]
ExpressionOperation = Literal[
    "SUM", "MEAN", "MEDIAN", "MIN", "MAX", "COUNT", "COUNT_DISTINCT",
    "STDDEV", "RANGE",
]
TimeGranularity = Literal["YEAR", "QUARTER", "MONTH", "WEEK", "DAY", "HOUR"]
EvidenceTier = Literal["EXACT", "SEMANTIC", "WEAK", "INELIGIBLE"]


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class DatasetColumnKnowledge(StrictFrozenModel):
    name: str = Field(min_length=1)
    physical_dtype: str | None = None
    semantic_type: str | None = None
    semantic_role: str | None = None
    value_function: str | None = None
    numeric_meaning: str | None = None
    concept_description: str | None = None
    reasoning_summary: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    referenced_entity: str | None = None
    entity_alignment: str | None = None
    quality_metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetKnowledge(StrictFrozenModel):
    dataset_id: str | None = None
    dataset_description: str | None = None
    row_entity: str | None = None
    row_entity_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    columns: list[DatasetColumnKnowledge] = Field(default_factory=list)
    quality: dict[str, Any] = Field(default_factory=dict)
    metadata_availability: dict[str, bool] = Field(default_factory=dict)

    def get_column(self, name: str) -> DatasetColumnKnowledge | None:
        folded = name.casefold()
        return next((column for column in self.columns if column.name.casefold() == folded), None)


class AnalyticalExpression(StrictFrozenModel):
    kind: ExpressionKind
    column: str | None = None
    operation: ExpressionOperation | None = None
    time_granularity: TimeGranularity | None = None

    @model_validator(mode="after")
    def validate_expression(self) -> "AnalyticalExpression":
        column_required = self.kind not in {"COUNT_ROWS"}
        if column_required and self.column is None:
            raise ValueError(f"{self.kind} requires a column")
        if not column_required and self.column is not None:
            raise ValueError("COUNT_ROWS cannot contain a column")
        operation_required = self.kind in {"AGGREGATION", "STATISTIC"}
        if operation_required and self.operation is None:
            raise ValueError(f"{self.kind} requires an operation")
        if not operation_required and self.operation is not None:
            if not (self.kind == "COUNT_DISTINCT" and self.operation == "COUNT_DISTINCT"):
                raise ValueError(f"{self.kind} cannot contain an operation")
        if self.kind == "TIME_BUCKET" and self.time_granularity is None:
            raise ValueError("TIME_BUCKET requires time_granularity")
        if self.kind != "TIME_BUCKET" and self.time_granularity is not None:
            raise ValueError("only TIME_BUCKET may contain time_granularity")
        return self


class GroundingCandidate(StrictFrozenModel):
    expression: AnalyticalExpression
    semantic_score: float = Field(ge=0.0, le=1.0)
    role_compatible: bool
    type_compatible: bool
    evidence_tier: EvidenceTier
    evidence: list[str] = Field(default_factory=list)
    pipeline1_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class GroundedConcept(StrictFrozenModel):
    source_concept: str
    slot_type: GroundingSlot
    status: GroundingStatus
    resolved_expression: AnalyticalExpression | None = None
    candidates: list[GroundingCandidate] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    clarification_reason: str | None = None

    @model_validator(mode="after")
    def validate_resolution(self) -> "GroundedConcept":
        if self.status == "RESOLVED" and self.resolved_expression is None:
            raise ValueError("RESOLVED grounding requires resolved_expression")
        if self.status != "RESOLVED" and self.resolved_expression is not None:
            raise ValueError("non-resolved grounding cannot contain resolved_expression")
        return self


class GroundingResult(StrictFrozenModel):
    status: GroundingStatus
    measures: list[GroundedConcept] = Field(default_factory=list)
    dimensions: list[GroundedConcept] = Field(default_factory=list)
    filters: list[GroundedConcept] = Field(default_factory=list)
    temporal: list[GroundedConcept] = Field(default_factory=list)
    ambiguities: list[GroundedConcept] = Field(default_factory=list)
    unresolved: list[GroundedConcept] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)


__all__ = [
    "AnalyticalExpression",
    "DatasetColumnKnowledge",
    "DatasetKnowledge",
    "EvidenceTier",
    "ExpressionKind",
    "ExpressionOperation",
    "GroundedConcept",
    "GroundingCandidate",
    "GroundingResult",
    "GroundingSlot",
    "GroundingStatus",
    "TimeGranularity",
]
