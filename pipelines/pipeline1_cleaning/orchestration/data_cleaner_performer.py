"""Deterministic execution of compiled cleaning instructions."""

from __future__ import annotations

import pandas as pd

from pipelines.pipeline1_cleaning.decisions.decision_schema import (
    CleaningInstruction,
    ExecutionResult,
)
from pipelines.pipeline1_cleaning.orchestration.action_registry import get_action
from pipelines.pipeline1_cleaning.profiling.column_profiler import make_serializable
from pipelines.pipeline1_cleaning.techniques.missing_values import prepare_imputation, missingness_assessment


def _changed_cells(before: pd.DataFrame, after: pd.DataFrame, cap: int = 5):
    changed_count = 0
    examples = []
    for column in before.columns:
        left = before[column].astype(object).where(before[column].notna(), "")
        right = after[column].astype(object).where(after[column].notna(), "")
        equal = (left.eq(right) & before[column].notna() & after[column].notna()) | (
            before[column].isna() & after[column].isna()
        )
        if before[column].dtype != after[column].dtype:
            equal = before[column].isna() & after[column].isna()
        for index in before.index[~equal.fillna(False)]:
            changed_count += 1
            if len(examples) < cap:
                examples.append({
                    "row_index": make_serializable(index),
                    "column": str(column),
                    "before": make_serializable(before.at[index, column]),
                    "after": make_serializable(after.at[index, column]),
                })
    return changed_count, examples


def execute_instruction(
    dataframe: pd.DataFrame,
    instruction: CleaningInstruction,
) -> tuple[pd.DataFrame, ExecutionResult]:
    """Execute against a deep copy so the caller's dataframe remains untouched."""
    working = dataframe.copy(deep=True)
    before = working.copy(deep=True)
    audit_metadata = {}
    if instruction.action_id == "impute_missing" and len(instruction.target_columns) == 1:
        column = instruction.target_columns[0]
        if column in before.columns:
            audit_metadata = {"strategy": instruction.parameters.get("strategy"),
                              **missingness_assessment(before[column])}
    try:
        handler = get_action(instruction.action_id)
        cleaned = handler(
            working, list(instruction.target_columns), dict(instruction.parameters)
        )
        if not isinstance(cleaned, pd.DataFrame):
            raise TypeError("Cleaning actions must return a pandas DataFrame.")
        if instruction.action_id == "impute_missing":
            _, audit_metadata = prepare_imputation(before[instruction.target_columns[0]], instruction.parameters)
    except Exception as exc:
        return before, ExecutionResult(
            instruction_id=instruction.instruction_id,
            source_decision_id=instruction.source_decision_id,
            source_issue_ids=instruction.source_issue_ids,
            action_id=instruction.action_id,
            target_columns=instruction.target_columns,
            status="failed", changed=False, changed_cell_count=0,
            audit_metadata=audit_metadata,
            rows_before=len(before), rows_after=len(before),
            columns_before=len(before.columns), columns_after=len(before.columns),
            message="Instruction execution failed; the input copy was preserved.",
            error=f"{type(exc).__name__}: {exc}",
        )

    changed_cell_count, change_examples = _changed_cells(before, cleaned)
    changed = not cleaned.equals(before)
    result = ExecutionResult(
        instruction_id=instruction.instruction_id,
        source_decision_id=instruction.source_decision_id,
        source_issue_ids=instruction.source_issue_ids,
        action_id=instruction.action_id,
        target_columns=instruction.target_columns,
        status="success",
        changed=changed,
        changed_cell_count=changed_cell_count,
        audit_metadata=audit_metadata,
        change_examples=change_examples,
        rows_before=len(before),
        rows_after=len(cleaned),
        columns_before=len(before.columns),
        columns_after=len(cleaned.columns),
        message=(
            "No changes were requested or applied."
            if instruction.action_id == "no_op"
            else "Instruction executed successfully."
        ),
    )
    return cleaned, result
