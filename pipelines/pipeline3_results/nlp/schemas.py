"""Shared, execution-free NLP contracts for Pipeline 3.

These models describe language interpretation only.  They deliberately do not
import pandas, inspect datasets, execute aggregations, or construct charts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from pipelines.pipeline3_results.nlp.chart_types import RequestedChart


QueryIntent = Literal["VISUALIZATION", "INSIGHT", "UNKNOWN"]
IntentConfidence = Literal["HIGH", "MEDIUM", "LOW"]
Aggregation = Literal["SUM", "MEAN", "COUNT", "MIN", "MAX", "MEDIAN", "NONE"]
ComparisonOperation = Literal["HIGHEST", "LOWEST", "TOP_N", "BOTTOM_N", "NONE"]
InterpretationStatus = Literal["RESOLVED", "NEEDS_CLARIFICATION", "INVALID"]
# Backward-compatible name for the one canonical production chart vocabulary.
SharedChartType = RequestedChart
SharedChartVariant = Literal["DOUGHNUT"]
FilterOperator = Literal[
    "EQUALS", "NOT_EQUALS", "GREATER_THAN", "GREATER_THAN_OR_EQUAL",
    "LESS_THAN", "LESS_THAN_OR_EQUAL", "CONTAINS", "UNKNOWN",
]
TermKind = Literal["MEASURE", "DIMENSION", "TIME", "FILTER", "RELATIONSHIP"]
FailureCode = Literal[
    "UNRESOLVED_CONCEPT",
    "AMBIGUOUS_CONCEPT",
    "UNSUPPORTED_OPERATION",
    "UNSUPPORTED_CHART",
    "INVALID_COMBINATION",
    "UNSUPPORTED_ANALYSIS",
]

ALL_QUERY_INTENTS: tuple[QueryIntent, ...] = (
    "VISUALIZATION", "INSIGHT", "UNKNOWN",
)
ALL_AGGREGATIONS: tuple[Aggregation, ...] = (
    "SUM", "MEAN", "COUNT", "MIN", "MAX", "MEDIAN", "NONE",
)
ALL_COMPARISON_OPERATIONS: tuple[ComparisonOperation, ...] = (
    "HIGHEST", "LOWEST", "TOP_N", "BOTTOM_N", "NONE",
)
ALL_INTERPRETATION_STATUSES: tuple[InterpretationStatus, ...] = (
    "RESOLVED", "NEEDS_CLARIFICATION", "INVALID",
)
ALL_SHARED_CHART_TYPES: tuple[SharedChartType, ...] = (
    "BAR", "LINE", "SCATTER", "HISTOGRAM", "COUNT_BAR", "PIE", "DOUGHNUT",
)


class QueryFilter(BaseModel):
    """An extracted filter expression; it is descriptive, never executable."""

    model_config = ConfigDict(frozen=True)

    raw_text: str
    column_term: str | None = None
    operator: FilterOperator = "UNKNOWN"
    value_term: str | None = None
    resolved_column: str | None = None


class IntentClassification(BaseModel):
    """Explainable deterministic routing decision for one user request."""

    model_config = ConfigDict(frozen=True)

    intent: QueryIntent
    confidence: IntentConfidence
    signals: tuple[str, ...] = ()
    context_used: bool = False


class ExtractedEntities(BaseModel):
    """Language concepts extracted before any dataset-column resolution."""

    model_config = ConfigDict(frozen=True)

    measure_terms: list[str] = Field(default_factory=list)
    dimension_terms: list[str] = Field(default_factory=list)
    aggregation: Aggregation = "NONE"
    comparison: ComparisonOperation = "NONE"
    requested_chart: SharedChartType | None = None
    requested_chart_variant: SharedChartVariant | None = None
    unsupported_chart_term: str | None = None
    filters: list[QueryFilter] = Field(default_factory=list)
    time_terms: list[str] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    signals: tuple[str, ...] = ()


class ColumnCandidate(BaseModel):
    """One possible dataset column and the deterministic evidence supporting it."""

    model_config = ConfigDict(frozen=True)

    column_name: str
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_sources: list[str] = Field(default_factory=list)
    semantic_evidence: dict[str, JsonValue] = Field(default_factory=dict)
    analytical_role: str | None = None


class TermResolution(BaseModel):
    """Resolution state for one unresolved human-language term."""

    model_config = ConfigDict(frozen=True)

    term: str
    kind: TermKind
    candidates: list[ColumnCandidate] = Field(default_factory=list)
    resolved_column: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None


class AmbiguityDecision(BaseModel):
    """Deterministic interpretation decision for one required query term."""

    model_config = ConfigDict(frozen=True)

    term: str
    kind: TermKind
    status: InterpretationStatus
    required: bool = True
    resolved_column: str | None = None
    choices: list[str] = Field(default_factory=list)
    reason: str
    signals: tuple[str, ...] = ()


class AmbiguityResult(BaseModel):
    """Overall ambiguity status and the auditable per-term decisions behind it."""

    model_config = ConfigDict(frozen=True)

    status: InterpretationStatus
    decisions: list[AmbiguityDecision] = Field(default_factory=list)
    reason: str
    signals: tuple[str, ...] = ()


class StructuredQuery(BaseModel):
    """Normalized query preserving language terms and any later resolutions."""

    model_config = ConfigDict(frozen=True)

    raw_text: str
    normalized_text: str | None = None
    intent: QueryIntent = "UNKNOWN"
    measure_terms: list[str] = Field(default_factory=list)
    dimension_terms: list[str] = Field(default_factory=list)
    aggregation: Aggregation = "NONE"
    comparison: ComparisonOperation = "NONE"
    requested_chart: SharedChartType | None = None
    requested_chart_variant: SharedChartVariant | None = None
    unsupported_chart_term: str | None = None
    filters: list[QueryFilter] = Field(default_factory=list)
    time_terms: list[str] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1)
    resolved_measures: list[str] = Field(default_factory=list)
    resolved_dimensions: list[str] = Field(default_factory=list)
    resolved_time_columns: list[str] = Field(default_factory=list)
    resolutions: list[TermResolution] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class QueryInterpretation(BaseModel):
    """Final shared-NLP interpretation handed to a deterministic subsystem."""

    model_config = ConfigDict(frozen=True)

    status: InterpretationStatus
    query: StructuredQuery
    ambiguity_decisions: list[AmbiguityDecision] = Field(default_factory=list)
    intent_confidence: IntentConfidence | None = None
    context_used: bool = False
    message: str | None = None
    choices: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    analytical_plan: dict[str, JsonValue] | None = None


__all__ = [
    "ALL_AGGREGATIONS",
    "ALL_COMPARISON_OPERATIONS",
    "ALL_INTERPRETATION_STATUSES",
    "ALL_QUERY_INTENTS",
    "ALL_SHARED_CHART_TYPES",
    "Aggregation",
    "AmbiguityDecision",
    "AmbiguityResult",
    "ColumnCandidate",
    "ComparisonOperation",
    "ExtractedEntities",
    "FilterOperator",
    "FailureCode",
    "InterpretationStatus",
    "IntentClassification",
    "IntentConfidence",
    "QueryFilter",
    "QueryIntent",
    "QueryInterpretation",
    "RequestedChart",
    "SharedChartType",
    "SharedChartVariant",
    "StructuredQuery",
    "TermKind",
    "TermResolution",
]
