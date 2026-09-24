"""HTTP-facing schemas for the first LYNX application integration stages."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


RunState = Literal["queued", "running", "complete", "failed"]
PredictionState = Literal["queued", "running", "complete", "failed"]


class ControlledError(BaseModel):
    code: str
    message: str


class CreateRunResponse(BaseModel):
    run_id: str
    state: Literal["queued"] = "queued"
    status_url: str


class RunStatusResponse(BaseModel):
    run_id: str
    state: RunState
    stage: str
    message: str
    created_at: str
    updated_at: str
    original_filename: str
    stored_extension: str
    pipeline1_internal_run_id: str | None = None
    error: ControlledError | None = None


class PreparedDatasetSummary(BaseModel):
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    target_columns: list[str]


class PredictionEligibility(BaseModel):
    eligible: bool
    reasons: list[str]
    cleaned_dataset_sha256_verified: bool


class Pipeline1ResultResponse(BaseModel):
    run_id: str
    state: Literal["complete"] = "complete"
    pipeline1_internal_run_id: str
    pipeline1_report: dict[str, Any]
    prepared_dataset: PreparedDatasetSummary
    privacy_clearance_passed: bool
    prediction_eligibility: PredictionEligibility


class PredictionCreateRequest(BaseModel):
    target_column: str

    @field_validator("target_column")
    @classmethod
    def target_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("target_column must be a non-empty string")
        return value


class PredictionCreateResponse(BaseModel):
    run_id: str
    prediction_id: str
    state: Literal["queued"] = "queued"
    status_url: str


class PredictionStatusResponse(BaseModel):
    run_id: str
    prediction_id: str
    state: PredictionState
    stage: str
    target_column: str
    created_at: str
    updated_at: str
    error: ControlledError | None = None


class FrontendTaskResolution(BaseModel):
    status: str
    task_type: Literal["classification", "regression"]
    explanation: str


class FrontendPredictionDataset(BaseModel):
    name: str | None = None
    total_rows: int | None = Field(default=None, ge=0)
    training_rows: int | None = Field(default=None, ge=0)
    test_rows: int | None = Field(default=None, ge=0)
    feature_count: int | None = Field(default=None, ge=0)
    target_missing_count: int | None = Field(default=None, ge=0)


class FrontendPredictionSummary(BaseModel):
    available: bool
    row_count: int | None = Field(default=None, ge=0)
    has_probabilities: bool
    confidence_available: bool


class FrontendPredictionRuntime(BaseModel):
    train_runtime_seconds: float | None = Field(default=None, ge=0)
    inference_runtime_seconds: float | None = Field(default=None, ge=0)


class FrontendPredictionResult(BaseModel):
    status: Literal["complete"] = "complete"
    prediction_id: str
    target: str
    task_type: Literal["classification", "regression"]
    task_resolution: FrontendTaskResolution
    dataset: FrontendPredictionDataset
    metrics: dict[str, Any]
    predictions: FrontendPredictionSummary
    runtime: FrontendPredictionRuntime


class PredictionRowsResponse(BaseModel):
    prediction_id: str
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=200)
    returned: int = Field(ge=0)
    total: int = Field(ge=0)
    rows: list[dict[str, Any]]


__all__ = [
    "ControlledError",
    "CreateRunResponse",
    "FrontendPredictionResult",
    "Pipeline1ResultResponse",
    "PredictionCreateRequest",
    "PredictionCreateResponse",
    "PredictionEligibility",
    "PredictionRowsResponse",
    "PredictionStatusResponse",
    "PreparedDatasetSummary",
    "RunStatusResponse",
]
