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


class RawDataPreview(BaseModel):
    available: bool
    unavailable_reason: str | None = None
    source_stage: Literal["original_ingestion"]
    row_limit: int = Field(ge=0, le=10)
    column_limit: int = Field(ge=1, le=100)
    total_row_count: int = Field(ge=0)
    total_column_count: int = Field(ge=0)
    columns_truncated: bool
    columns: list[str]
    rows: list[list[Any]]
    returned_row_count: int = Field(ge=0, le=10)
    privacy_redacted_columns: list[str]


class DatasetOverviewMetadata(BaseModel):
    memory_footprint_bytes: int | None = Field(default=None, ge=0)
    memory_footprint_stage: Literal["original_ingestion"]
    raw_data_preview: RawDataPreview


class ProfilingActionRecord(BaseModel):
    decision_id: str
    action_id: str
    decision_type: Literal[
        "action", "no_action", "flag_for_review", "insufficient_evidence"
    ]
    status: Literal[
        "proposed", "applied", "skipped", "not_required",
        "requires_review", "failed", "unavailable",
    ]
    rationale: str | None = None
    evidence: dict[str, Any]
    requires_review: bool
    execution_status: Literal["success", "failed"] | None = None
    changed: bool | None = None
    changed_cell_count: int | None = Field(default=None, ge=0)


class ProfilingColumnRecord(BaseModel):
    column_index: int = Field(ge=0)
    column_name: str
    physical_dtype: str | None = None
    semantic_role: str | None = None
    semantic_type: str | None = None
    meaning: str | None = None
    semantic_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    non_null_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    missing_percentage: float = Field(ge=0.0, le=100.0)
    unique_count: int = Field(ge=0)
    primary_action: ProfilingActionRecord | None = None
    actions: list[ProfilingActionRecord]
    status: Literal[
        "proposed", "applied", "skipped", "not_required",
        "requires_review", "failed", "unavailable",
    ]
    decision_explanation: str | None = None
    explanation_source: Literal[
        "cleaning_decision", "semantic_reasoning", "semantic_description", "unavailable"
    ]
    observed_examples: list[Any]
    privacy_restricted: bool


class ProfilingManifest(BaseModel):
    dataset_name: str
    profiled_at: str | None = None
    profile_status: Literal["completed", "partial", "unavailable", "failed"]
    total_columns: int = Field(ge=0)
    semantic_roles_detected: list[str]
    semantic_role_count: int = Field(ge=0)
    columns_requiring_cleaning: int = Field(ge=0)
    columns: list[ProfilingColumnRecord]


class Pipeline1ResultResponse(BaseModel):
    run_id: str
    state: Literal["complete"] = "complete"
    pipeline1_internal_run_id: str
    pipeline1_report: dict[str, Any]
    prepared_dataset: PreparedDatasetSummary
    privacy_clearance_passed: bool
    prediction_eligibility: PredictionEligibility
    dataset_overview: DatasetOverviewMetadata | None = None
    profiling_manifest: ProfilingManifest | None = None


class DatasetPageResponse(BaseModel):
    run_id: str
    dataset_name: str
    version: Literal["original", "cleaned"]
    columns: list[str]
    rows: list[list[Any]]
    row_numbers: list[int]
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    total_rows: int = Field(ge=0)
    filtered_rows: int = Field(ge=0)
    privacy_redacted_columns: list[str]
    change_highlighting_available: bool = False
    change_highlighting_reason: str | None = None


class PredictionCreateRequest(BaseModel):
    target_column: str

    @field_validator("target_column")
    @classmethod
    def target_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("target_column must be a non-empty string")
        return value


class InsightQueryRequest(BaseModel):
    """Minimal public request contract for deterministic Insights."""

    request_text: str

    @field_validator("request_text")
    @classmethod
    def request_text_must_be_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("request_text must be a non-empty string")
        return stripped


class PredictionCreateResponse(BaseModel):
    run_id: str
    prediction_id: str
    state: Literal["queued"] = "queued"
    status_url: str


class PredictionTargetResolutionResponse(BaseModel):
    target_column: str
    status: Literal["resolved", "unsupported"]
    task_type: Literal["classification", "regression"] | None = None
    classification_type: str | None = None
    explanation: str


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
    classification_type: str | None = None
    completed_at: str | None = None
    task_resolution: FrontendTaskResolution
    dataset: FrontendPredictionDataset
    metrics: dict[str, Any]
    feature_importance: dict[str, Any]
    class_labels: list[Any] | None = None
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
    "DatasetOverviewMetadata",
    "FrontendPredictionResult",
    "InsightQueryRequest",
    "Pipeline1ResultResponse",
    "PredictionCreateRequest",
    "PredictionCreateResponse",
    "PredictionTargetResolutionResponse",
    "PredictionEligibility",
    "ProfilingActionRecord",
    "ProfilingColumnRecord",
    "ProfilingManifest",
    "PredictionRowsResponse",
    "PredictionStatusResponse",
    "PreparedDatasetSummary",
    "RawDataPreview",
    "RunStatusResponse",
]
