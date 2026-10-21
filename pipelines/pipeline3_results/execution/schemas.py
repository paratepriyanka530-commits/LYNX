"""Canonical verified-result contracts for deterministic analytical execution."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


ResultStatus = Literal["READY", "EMPTY", "PARTIAL", "UNSUPPORTED", "INVALID", "EXECUTION_ERROR"]
ResultType = Literal[
    "SCALAR", "TABLE", "SERIES", "DISTRIBUTION", "RELATIONSHIP",
    "STATISTIC", "EXPLORATORY_PLACEHOLDER",
]
VerificationStatus = Literal["VERIFIED", "VERIFIED_WITH_WARNINGS", "FAILED", "NOT_RUN"]
ExecutionErrorCode = Literal[
    "UNSUPPORTED_OPERATION", "INVALID_PLAN", "MISSING_COLUMN", "TYPE_MISMATCH",
    "EMPTY_RESULT", "TEMPORAL_PARSE_ERROR", "NON_FINITE_RESULT",
    "NEGATIVE_CONTRIBUTION", "ZERO_DENOMINATOR", "EXECUTION_ERROR",
]


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ResultField(StrictFrozenModel):
    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    role: Literal["DIMENSION", "MEASURE", "TEMPORAL", "VALUE", "COUNT", "SHARE", "BIN"]


class ResultSchema(StrictFrozenModel):
    fields: list[ResultField] = Field(default_factory=list)


class ExecutionCoverage(StrictFrozenModel):
    source_rows: int = Field(ge=0)
    filtered_rows: int = Field(ge=0)
    usable_rows: int = Field(ge=0)
    excluded_null_rows: int = Field(default=0, ge=0)
    excluded_invalid_rows: int = Field(default=0, ge=0)


class ExecutionMetadata(StrictFrozenModel):
    backend: str
    operation: str
    expressions: list[dict[str, JsonValue]] = Field(default_factory=list)
    groupings: list[dict[str, JsonValue]] = Field(default_factory=list)
    filters_applied: list[dict[str, JsonValue]] = Field(default_factory=list)
    sort: list[dict[str, JsonValue]] = Field(default_factory=list)
    temporal_bucket: dict[str, JsonValue] | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ExecutionError(StrictFrozenModel):
    code: ExecutionErrorCode
    message: str


class ResultVerification(StrictFrozenModel):
    status: VerificationStatus = "NOT_RUN"
    checks: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class VerifiedResult(StrictFrozenModel):
    result_id: str = Field(pattern=r"^result_[0-9a-f]{16}$")
    plan_id: str
    status: ResultStatus
    result_type: ResultType
    scalar: JsonValue | None = None
    rows: list[dict[str, JsonValue]] = Field(default_factory=list)
    schema_: ResultSchema = Field(alias="schema")
    coverage: ExecutionCoverage
    execution_metadata: ExecutionMetadata
    warnings: list[str] = Field(default_factory=list)
    errors: list[ExecutionError] = Field(default_factory=list)
    verification: ResultVerification = Field(default_factory=ResultVerification)

    @model_validator(mode="after")
    def validate_shape(self) -> "VerifiedResult":
        if self.result_type in {"SCALAR", "STATISTIC"} and self.status == "READY" and self.scalar is None:
            raise ValueError(f"{self.result_type} READY result requires scalar")
        if self.result_type not in {"SCALAR", "STATISTIC"} and self.scalar is not None:
            raise ValueError(f"{self.result_type} cannot contain scalar")
        return self

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )


__all__ = [
    "ExecutionCoverage", "ExecutionError", "ExecutionErrorCode",
    "ExecutionMetadata", "ResultField", "ResultSchema", "ResultStatus",
    "ResultType", "ResultVerification", "VerificationStatus", "VerifiedResult",
]
