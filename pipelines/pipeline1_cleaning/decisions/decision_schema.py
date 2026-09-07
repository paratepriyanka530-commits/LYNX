"""Minimal contracts shared by the Pipeline 1 cleaning path."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, model_validator


DecisionOutcome = Literal[
    "action",
    "no_action",
    "flag_for_review",
    "insufficient_evidence",
]

ProposedAction = Literal[
    "no_op", "trim_whitespace", "normalize_categorical_labels",
    "parse_dates", "coerce_dtype", "impute_missing",
    "add_missing_indicator", "drop_duplicate_rows", "flag_rows",
]


class CleaningDecision(BaseModel):
    decision_id: str = Field(min_length=1)
    source_issue_ids: List[str] = Field(min_length=1)
    target_columns: List[str] = Field(default_factory=list)
    decision_type: DecisionOutcome
    proposed_action: ProposedAction
    parameters: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    evidence_summary: Dict[str, Any] = Field(default_factory=dict)
    semantic_basis: Dict[str, Any] = Field(default_factory=dict)
    risk_level: Literal["low", "medium", "high"]
    requires_review: bool
    decision_source: Literal["deterministic_tier1", "llm_tier2", "fallback"]

    @model_validator(mode="after")
    def validate_action(self):
        if self.decision_type == "no_action" and self.proposed_action != "no_op":
            raise ValueError("no_action decisions must propose no_op.")
        if self.decision_type == "flag_for_review" and self.proposed_action != "flag_rows":
            raise ValueError("flag_for_review decisions must propose flag_rows.")
        if self.decision_type == "insufficient_evidence" and self.proposed_action != "no_op":
            raise ValueError("insufficient_evidence decisions must propose no_op.")
        return self


class CleaningInstruction(BaseModel):
    instruction_id: str = Field(min_length=1)
    source_decision_id: str = Field(min_length=1)
    source_issue_ids: List[str] = Field(min_length=1)
    action_id: str = Field(min_length=1)
    target_columns: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    instruction_id: str
    source_decision_id: str
    source_issue_ids: List[str] = Field(min_length=1)
    action_id: str
    target_columns: List[str] = Field(default_factory=list)
    status: Literal["success", "failed"]
    changed: bool
    changed_cell_count: int = Field(ge=0)
    change_examples: List[Dict[str, Any]] = Field(default_factory=list)
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    message: str
    error: str | None = None
    audit_metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    passed: bool
    no_unexpected_changes: bool
    dataframe_equal: bool
    row_count_equal: bool
    column_count_equal: bool
    missingness_equal: bool
    duplicate_row_count_equal: bool
    physical_dtypes_equal: bool
    cardinality_equal: bool
    unrelated_columns_equal: bool = True
    targeted_whitespace_improved: bool = True
    targeted_case_variants_improved: bool = True
    only_approved_value_changes: bool = True
    no_unexpected_categories: bool = True
    targeted_date_formats_improved: bool = True
    only_approved_date_changes: bool = True
    expected_dtype_changes_only: bool = True
    only_approved_numeric_changes: bool = True
    null_positions_equal: bool = True
    targeted_mixed_numeric_improved: bool = True
    only_approved_imputation_changes: bool = True
    expected_missingness_changes_only: bool = True
    imputed_cell_count: int = Field(default=0, ge=0)
    imputation_audit: List[Dict[str, Any]] = Field(default_factory=list)
    newly_introduced_null_count: int = Field(default=0, ge=0)
    changed_cell_count: int = Field(default=0, ge=0)
    before_issue_counts: Dict[str, Any] = Field(default_factory=dict)
    after_issue_counts: Dict[str, Any] = Field(default_factory=dict)
    before_summary: Dict[str, Any]
    after_summary: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)


class Pipeline1Report(BaseModel):
    pipeline_version: str
    run_id: str
    status: Literal["success", "failed"]
    original_dataset: Dict[str, Any]
    before_profile: Dict[str, Any]
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    quality_summary: Dict[str, Any] = Field(default_factory=dict)
    semantic_profile: Dict[str, Any] = Field(default_factory=dict)
    decision_summary: Dict[str, Any] = Field(default_factory=dict)
    decisions: List[CleaningDecision] = Field(default_factory=list)
    instructions: List[CleaningInstruction] = Field(default_factory=list)
    executions: List[ExecutionResult] = Field(default_factory=list)
    validation: ValidationReport
    warnings: List[str] = Field(default_factory=list)
    output_artifacts: Dict[str, Any]
