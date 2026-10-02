"""Strict contracts for the pretrained-first NLP comparison prototype."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ExperimentalIntent = Literal["INSIGHT", "VISUALIZATION", "UNKNOWN"]
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
ExperimentalAggregation = Literal[
    "SUM", "MEAN", "COUNT", "MIN", "MAX", "MEDIAN", "NONE",
]
ExperimentalComparisonOperation = Literal[
    "HIGHEST", "LOWEST", "TOP_N", "BOTTOM_N", "NONE",
]
ExperimentalChartType = Literal[
    "BAR", "LINE", "SCATTER", "HISTOGRAM", "COUNT_BAR", "PIE", "DOUGHNUT",
]
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
TimeGranularity = Literal[
    "YEAR", "QUARTER", "MONTH", "WEEK", "DAY", "HOUR", "UNSPECIFIED",
]
GroundingStatus = Literal[
    "RESOLVED", "NEEDS_CLARIFICATION", "INVALID", "UNSUPPORTED",
]


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class SemanticFilter(StrictFrozenModel):
    """A semantic restriction that deliberately contains no physical column."""

    field_concept: str = Field(min_length=1)
    operator: SemanticFilterOperator
    value_concept: str | None


class SemanticTimeExpression(StrictFrozenModel):
    """Semantic time request, prior to resolving a temporal column."""

    expression: str = Field(min_length=1)
    granularity: TimeGranularity


class AnalyticalInterpretation(StrictFrozenModel):
    """One model-produced interpretation of analytical meaning only."""

    intent: ExperimentalIntent = Field(
        description="Whether the user asks for an insight or explicitly asks for a chart."
    )
    analytical_goal: AnalyticalGoal = Field(
        description="The analytical operation; ranking language requires RANK_GROUPS, TOP_N, or BOTTOM_N."
    )
    measure_concepts: list[str] = Field(
        description="General quantities to analyze, expressed as semantic phrases, never column names.",
    )
    dimension_concepts: list[str] = Field(
        description="General entities or groups to compare, expressed semantically, never column names.",
    )
    aggregation: ExperimentalAggregation = Field(
        description="How the measure is summarized. Revenue across records normally uses SUM.",
    )
    comparison: ExperimentalComparisonOperation = Field(
        description="Highest/lowest/ranked direction explicitly expressed by the user.",
    )
    limit: int | None = Field(ge=1)
    requested_chart: ExperimentalChartType | None
    filters: list[SemanticFilter]
    time_expression: SemanticTimeExpression | None
    requires_clarification: bool = Field(
        description="True when a required metric, entity, grouping, or meaning is underspecified."
    )
    clarification_reason: str | None
    unsupported: bool
    unsupported_reason: str | None
    reasoning_summary: str = Field(min_length=1, max_length=500)

    @field_validator("measure_concepts", "dimension_concepts")
    @classmethod
    def _concepts_are_nonempty_and_unique(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            clean = value.strip()
            if not clean:
                raise ValueError("semantic concepts must not be empty")
            key = clean.casefold()
            if key not in seen:
                seen.add(key)
                normalized.append(clean)
        return normalized

    @model_validator(mode="after")
    def _validate_semantic_state(self) -> "AnalyticalInterpretation":
        if self.requires_clarification and not self.clarification_reason:
            raise ValueError("clarification_reason is required when clarification is true")
        if not self.requires_clarification and self.clarification_reason is not None:
            raise ValueError("clarification_reason must be null when clarification is false")
        if self.unsupported and not self.unsupported_reason:
            raise ValueError("unsupported_reason is required when unsupported is true")
        if not self.unsupported and self.unsupported_reason is not None:
            raise ValueError("unsupported_reason must be null when unsupported is false")
        return self


class GroundedConcept(StrictFrozenModel):
    concept: str
    kind: Literal["MEASURE", "DIMENSION", "TIME", "FILTER"]
    status: Literal["RESOLVED", "NEEDS_CLARIFICATION", "INVALID"]
    resolved_column: str | None = None
    candidate_columns: list[str] = Field(default_factory=list)
    reason: str
    evidence: dict[str, list[str]] = Field(default_factory=dict)


class GroundedInterpretation(StrictFrozenModel):
    status: GroundingStatus
    resolved_measures: list[str] = Field(default_factory=list)
    resolved_dimensions: list[str] = Field(default_factory=list)
    resolved_time_columns: list[str] = Field(default_factory=list)
    concepts: list[GroundedConcept] = Field(default_factory=list)
    reason: str


class ManualEvaluation(StrictFrozenModel):
    """Human labels; null means deliberately not scored yet."""

    INTENT_CORRECT: bool | None = None
    GOAL_CORRECT: bool | None = None
    MEASURE_CONCEPT_CORRECT: bool | None = None
    DIMENSION_CONCEPT_CORRECT: bool | None = None
    AGGREGATION_CORRECT: bool | None = None
    COMPARISON_CORRECT: bool | None = None
    SHOULD_CLARIFY_CORRECT: bool | None = None
    GROUNDING_CORRECT: bool | None = None


class ExperimentalComparison(StrictFrozenModel):
    query: str
    dataset_key: str
    deterministic: dict[str, Any]
    pretrained: AnalyticalInterpretation | None = None
    grounded_pretrained: GroundedInterpretation | None = None
    latency_seconds: float | None = Field(default=None, ge=0.0)
    provider_metadata: dict[str, Any] | None = None
    error: str | None = None
    manual_evaluation: ManualEvaluation = Field(default_factory=ManualEvaluation)


__all__ = [
    "AnalyticalGoal",
    "AnalyticalInterpretation",
    "ExperimentalAggregation",
    "ExperimentalChartType",
    "ExperimentalComparison",
    "ExperimentalComparisonOperation",
    "ExperimentalIntent",
    "GroundedConcept",
    "GroundedInterpretation",
    "GroundingStatus",
    "ManualEvaluation",
    "SemanticFilter",
    "SemanticTimeExpression",
]
