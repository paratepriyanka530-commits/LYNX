"""Provenance-aware contracts for language-understanding experiment V2."""

from __future__ import annotations

from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SourceType(str, Enum):
    EXPLICIT = "EXPLICIT"
    INFERRED = "INFERRED"
    MISSING = "MISSING"


IntentV2 = Literal["INSIGHT", "VISUALIZATION", "UNKNOWN"]
AnalyticalGoalV2 = Literal[
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
AggregationV2 = Literal["SUM", "MEAN", "COUNT", "MIN", "MAX", "MEDIAN", "NONE"]
ComparisonV2 = Literal["HIGHEST", "LOWEST", "TOP_N", "BOTTOM_N", "NONE"]
ChartTypeV2 = Literal[
    "BAR", "LINE", "SCATTER", "HISTOGRAM", "COUNT_BAR", "PIE", "DOUGHNUT",
]
FilterOperatorV2 = Literal[
    "EQUALS", "NOT_EQUALS", "GREATER_THAN", "GREATER_THAN_OR_EQUAL",
    "LESS_THAN", "LESS_THAN_OR_EQUAL", "CONTAINS", "BEFORE", "AFTER",
    "BETWEEN", "UNKNOWN",
]
TimeGranularityV2 = Literal[
    "YEAR", "QUARTER", "MONTH", "WEEK", "DAY", "HOUR", "UNSPECIFIED",
]
SlotName = Literal[
    "intent", "analytical_goal", "measure", "dimension", "aggregation",
    "comparison", "limit", "requested_chart", "filter", "time_expression",
]
AmbiguitySlot = Literal[
    "measure", "dimension", "aggregation", "comparison", "time_expression",
]
MissingRequiredSlot = Literal[
    "measure", "dimension", "aggregation", "comparison", "time_expression",
]


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


T = TypeVar("T")


class SemanticField(StrictFrozenModel, Generic[T]):
    """A value plus an auditable claim about where its meaning came from."""

    value: T | None
    source: SourceType
    evidence: str | None = Field(
        description="Exact phrase copied from the user query, or null when MISSING."
    )

    @model_validator(mode="after")
    def _validate_source_contract(self) -> "SemanticField[T]":
        if self.source == SourceType.MISSING:
            if self.value not in (None, []) or self.evidence is not None:
                raise ValueError("MISSING fields require an empty value and null evidence")
        elif self.value is None or not self.evidence:
            raise ValueError("EXPLICIT and INFERRED fields require value and evidence")
        return self


class SemanticFilterV2(StrictFrozenModel):
    field_concept: str = Field(min_length=1)
    operator: FilterOperatorV2
    value_concept: str | None


class SemanticTimeExpressionV2(StrictFrozenModel):
    expression: str = Field(min_length=1)
    granularity: TimeGranularityV2


class AmbiguousSlot(StrictFrozenModel):
    """A present concept with more than one plausible analytical meaning."""

    slot: AmbiguitySlot
    concept: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=300)


class MissingSlot(StrictFrozenModel):
    """Information required by the requested analysis but absent from the query."""

    slot: MissingRequiredSlot
    reason: str = Field(min_length=1, max_length=300)


class AnalyticalInterpretationV2(StrictFrozenModel):
    """Faithful language meaning, without data grounding or execution."""

    intent: SemanticField[IntentV2]
    analytical_goal: SemanticField[AnalyticalGoalV2]
    measures: SemanticField[list[str]] = Field(
        description="Free semantic measure phrases. Never physical dataframe columns."
    )
    dimensions: SemanticField[list[str]] = Field(
        description="Free semantic entity/group phrases. Never physical dataframe columns."
    )
    aggregation: SemanticField[AggregationV2]
    comparison: SemanticField[ComparisonV2]
    limit: SemanticField[int] = Field(
        description="MISSING unless the user explicitly states a numeric limit."
    )
    requested_chart: SemanticField[ChartTypeV2] = Field(
        description="MISSING unless the user explicitly names a chart type."
    )
    filters: SemanticField[list[SemanticFilterV2]]
    time_expression: SemanticField[SemanticTimeExpressionV2]
    ambiguities: list[AmbiguousSlot]
    missing_slots: list[MissingSlot]
    requires_clarification: bool
    clarification_question: str | None
    unsupported: bool
    unsupported_reason: str | None
    interpretation_summary: str = Field(
        min_length=1,
        max_length=300,
        description="Brief final interpretation rationale, never chain-of-thought.",
    )

    @field_validator("measures", "dimensions")
    @classmethod
    def _normalize_concept_lists(
        cls, field: SemanticField[list[str]]
    ) -> SemanticField[list[str]]:
        if field.value is None:
            return field
        if field.source == SourceType.MISSING and field.value == []:
            return field
        cleaned: list[str] = []
        seen: set[str] = set()
        for concept in field.value:
            value = concept.strip()
            if not value:
                raise ValueError("semantic concepts must not be empty")
            key = value.casefold()
            if key not in seen:
                seen.add(key)
                cleaned.append(value)
        if not cleaned:
            raise ValueError("present semantic concept lists must not be empty")
        return field.model_copy(update={"value": cleaned})

    @model_validator(mode="after")
    def _validate_interpretation_contract(self) -> "AnalyticalInterpretationV2":
        if self.limit.value is not None and self.limit.source != SourceType.EXPLICIT:
            raise ValueError("a numeric limit may only be EXPLICIT")
        if (
            self.requested_chart.value is not None
            and self.requested_chart.source != SourceType.EXPLICIT
        ):
            raise ValueError("a requested chart may only be EXPLICIT")
        if self.filters.value is not None and self.filters.source != SourceType.EXPLICIT:
            raise ValueError("filters may only be EXPLICIT")
        ambiguity_slots = {item.slot for item in self.ambiguities}
        missing_slots = {item.slot for item in self.missing_slots}
        overlap = ambiguity_slots & missing_slots
        if overlap:
            raise ValueError(f"slots cannot be both ambiguous and missing: {sorted(overlap)}")
        if self.requires_clarification:
            if not self.clarification_question:
                raise ValueError("clarification_question is required")
            if not self.missing_slots and not self.ambiguities:
                raise ValueError("clarification requires a missing or ambiguous slot")
        elif self.clarification_question is not None:
            raise ValueError("clarification_question must be null when clarification is false")
        if self.unsupported and not self.unsupported_reason:
            raise ValueError("unsupported_reason is required when unsupported is true")
        if not self.unsupported and self.unsupported_reason is not None:
            raise ValueError("unsupported_reason must be null when unsupported is false")
        return self


class ManualEvaluationV2(StrictFrozenModel):
    INTENT_CORRECT: bool | None = None
    GOAL_CORRECT: bool | None = None
    MEASURE_CORRECT: bool | None = None
    DIMENSION_CORRECT: bool | None = None
    AGGREGATION_CORRECT: bool | None = None
    COMPARISON_CORRECT: bool | None = None
    MISSING_INFO_CORRECT: bool | None = None
    NO_INVENTION: bool | None = None
    CLARIFICATION_CORRECT: bool | None = None


class ExperimentResultV2(StrictFrozenModel):
    query: str
    model_output: dict[str, Any] | None
    schema_validation_passed: bool
    structural_validation_passed: bool
    latency_seconds: float = Field(ge=0.0)
    provider_metadata: dict[str, object] | None
    error: str | None
    manual_evaluation: ManualEvaluationV2 = Field(default_factory=ManualEvaluationV2)


__all__ = [
    "AggregationV2",
    "AmbiguitySlot",
    "AmbiguousSlot",
    "AnalyticalGoalV2",
    "AnalyticalInterpretationV2",
    "ChartTypeV2",
    "ComparisonV2",
    "ExperimentResultV2",
    "IntentV2",
    "ManualEvaluationV2",
    "MissingRequiredSlot",
    "MissingSlot",
    "SemanticField",
    "SemanticFilterV2",
    "SemanticTimeExpressionV2",
    "SlotName",
    "SourceType",
]
