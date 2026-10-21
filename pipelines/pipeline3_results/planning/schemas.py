"""Canonical, execution-free analytical planning contracts."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from pipelines.pipeline3_results.grounding.schemas import (
    AnalyticalExpression,
    TimeGranularity,
)


PlanStatus = Literal["READY", "NEEDS_CLARIFICATION", "UNSUPPORTED", "INVALID"]


class PlanOperation(StrEnum):
    AGGREGATE = "AGGREGATE"
    GROUP_AGGREGATE = "GROUP_AGGREGATE"
    COUNT = "COUNT"
    RANK = "RANK"
    TOP_N = "TOP_N"
    BOTTOM_N = "BOTTOM_N"
    TREND = "TREND"
    DISTRIBUTION = "DISTRIBUTION"
    RELATIONSHIP = "RELATIONSHIP"
    STATISTIC = "STATISTIC"
    PART_TO_WHOLE = "PART_TO_WHOLE"
    EXPLORATORY = "EXPLORATORY"
GroupingKind = Literal["GROUP_BY", "TIME_BUCKET"]
FilterOperator = Literal[
    "EQUALS", "NOT_EQUALS", "IN", "NOT_IN", "GREATER_THAN",
    "GREATER_THAN_OR_EQUAL", "LESS_THAN", "LESS_THAN_OR_EQUAL", "BETWEEN",
    "BEFORE", "AFTER",
]
FilterValueType = Literal["STRING", "NUMBER", "BOOLEAN", "DATE", "DATETIME"]
SortTarget = Literal["MEASURE", "DIMENSION", "TEMPORAL"]
SortDirection = Literal["ASC", "DESC"]
Comparison = Literal["HIGHEST", "LOWEST", "TOP_N", "BOTTOM_N", "NONE"]
ScalarFilterValue = str | int | float | bool | date | datetime


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class GroupingExpression(StrictFrozenModel):
    kind: GroupingKind
    column: str = Field(min_length=1)
    time_granularity: TimeGranularity | None = None

    @model_validator(mode="after")
    def validate_grouping(self) -> "GroupingExpression":
        if self.kind == "TIME_BUCKET" and self.time_granularity is None:
            raise ValueError("TIME_BUCKET requires time_granularity")
        if self.kind == "GROUP_BY" and self.time_granularity is not None:
            raise ValueError("GROUP_BY cannot contain time_granularity")
        return self


class PlanFilter(StrictFrozenModel):
    column: str = Field(min_length=1)
    operator: FilterOperator
    value_type: FilterValueType
    value: ScalarFilterValue | list[ScalarFilterValue]

    @model_validator(mode="after")
    def validate_value_shape(self) -> "PlanFilter":
        multiple = isinstance(self.value, list)
        if self.operator == "BETWEEN" and (not multiple or len(self.value) != 2):
            raise ValueError("BETWEEN requires exactly two values")
        if self.operator in {"IN", "NOT_IN"} and (not multiple or not self.value):
            raise ValueError(f"{self.operator} requires a non-empty value list")
        if self.operator not in {"BETWEEN", "IN", "NOT_IN"} and multiple:
            raise ValueError(f"{self.operator} requires one scalar value")
        return self


class SortExpression(StrictFrozenModel):
    target: SortTarget
    direction: SortDirection
    measure_index: int | None = Field(default=None, ge=0)
    column: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "SortExpression":
        if self.target == "MEASURE" and self.measure_index is None:
            raise ValueError("measure sort requires measure_index")
        if self.target != "MEASURE" and self.column is None:
            raise ValueError("dimension/temporal sort requires column")
        if self.target == "MEASURE" and self.column is not None:
            raise ValueError("measure sort cannot contain a column")
        return self


class PlanProvenance(StrictFrozenModel):
    source_intent: dict[str, JsonValue] = Field(default_factory=dict)
    grounding_status: str
    resolved_expressions: list[dict[str, JsonValue]] = Field(default_factory=list)
    planner_decisions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnalyticalPlan(StrictFrozenModel):
    """Allow-listed description of a computation; never executable code."""

    plan_id: str = Field(pattern=r"^plan_[0-9a-f]{16}$")
    operation: PlanOperation
    measures: list[AnalyticalExpression] = Field(default_factory=list)
    dimensions: list[GroupingExpression] = Field(default_factory=list)
    filters: list[PlanFilter] = Field(default_factory=list)
    temporal: list[GroupingExpression] = Field(default_factory=list)
    sort: list[SortExpression] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1)
    comparison: Comparison = "NONE"
    provenance: PlanProvenance


class PlanBuildResult(StrictFrozenModel):
    status: PlanStatus
    plan: AnalyticalPlan | None = None
    message: str
    clarification_options: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    provenance: PlanProvenance | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "PlanBuildResult":
        if self.status == "READY" and self.plan is None:
            raise ValueError("READY requires a plan")
        if self.status != "READY" and self.plan is not None:
            raise ValueError("non-ready outcomes cannot contain a plan")
        return self


class PlanValidation(StrictFrozenModel):
    status: PlanStatus
    plan: AnalyticalPlan | None = None
    message: str
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "AnalyticalPlan", "Comparison", "FilterOperator", "FilterValueType",
    "GroupingExpression", "GroupingKind", "PlanBuildResult", "PlanFilter",
    "PlanOperation", "PlanProvenance", "PlanStatus", "PlanValidation",
    "SortDirection", "SortExpression", "SortTarget",
]
