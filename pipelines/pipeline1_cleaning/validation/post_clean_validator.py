"""Post-clean comparisons for deterministic Pipeline 1 execution."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from pipelines.pipeline1_cleaning.decisions.decision_schema import ValidationReport
from pipelines.pipeline1_cleaning.decisions.decision_schema import CleaningInstruction
from pipelines.pipeline1_cleaning.profiling.dataset_profiler import profile_dataset
from pipelines.pipeline1_cleaning.quality.issue_detector import detect_issues
from pipelines.pipeline1_cleaning.techniques.date_parsing import assess_dates, require_safe_dates
from pipelines.pipeline1_cleaning.techniques.numeric_coercion import TARGET_DTYPES, convert_numeric
from pipelines.pipeline1_cleaning.techniques.missing_values import prepare_imputation


def _issue_counts(dataframe, profile):
    issues = detect_issues(dataframe, profile)
    by_type = {}
    whitespace_by_column = {}
    case_variants_by_column = {}
    dates_by_column = {}
    numeric_by_column = {}
    for issue in issues:
        if issue.issue_type == "pii_candidate":
            continue
        by_type[issue.issue_type] = by_type.get(issue.issue_type, 0) + 1
        if issue.issue_type == "whitespace_issue":
            for column in issue.target_columns:
                whitespace_by_column[column] = issue.affected_count
        if issue.issue_type == "case_variant_candidate":
            for column in issue.target_columns:
                case_variants_by_column[column] = issue.affected_count
        if issue.issue_type == "inconsistent_date_format_candidate":
            for column in issue.target_columns:
                dates_by_column[column] = issue.affected_count
        if issue.issue_type == "mixed_numeric_representation_candidate":
            for column in issue.target_columns:
                numeric_by_column[column] = issue.affected_count
    return {
        "by_type": by_type,
        "whitespace_affected_cells_by_column": whitespace_by_column,
        "case_variant_affected_cells_by_column": case_variants_by_column,
        "date_format_affected_cells_by_column": dates_by_column,
        "mixed_numeric_affected_cells_by_column": numeric_by_column,
    }


def _profile_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    columns = profile["column_profiles"]
    return {
        "row_count": profile["shape"]["rows"],
        "column_count": profile["shape"]["columns"],
        "total_missing_cells": profile["missingness"]["total_missing_cells"],
        "duplicate_row_count": profile["duplicates"]["exact_duplicate_rows"],
        "physical_dtypes": {
            item["column_name"]: item["physical_dtype"] for item in columns
        },
        "cardinality": {
            item["column_name"]: item["uniqueness"]["unique_count"]
            for item in columns
        },
    }


def _validate_non_imputation(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    before_profile: Dict[str, Any],
    expect_no_changes: bool = False,
    modified_columns: list[str] | None = None,
    executed_instructions: list[CleaningInstruction] | None = None,
) -> tuple[ValidationReport, Dict[str, Any]]:
    modified_columns = modified_columns or []
    executed_instructions = executed_instructions or []
    after_profile = profile_dataset(cleaned)
    before = _profile_summary(before_profile)
    after = _profile_summary(after_profile)
    dataframe_equal = original.equals(cleaned)
    unrelated_columns = [
        column for column in original.columns if str(column) not in modified_columns
    ]
    unrelated_columns_equal = original[unrelated_columns].equals(cleaned[unrelated_columns])
    before_issue_counts = _issue_counts(original, before_profile)
    after_issue_counts = _issue_counts(cleaned, after_profile)
    before_whitespace = before_issue_counts["whitespace_affected_cells_by_column"]
    after_whitespace = after_issue_counts["whitespace_affected_cells_by_column"]
    normalization_instructions = [
        instruction for instruction in executed_instructions
        if instruction.action_id == "normalize_categorical_labels"
    ]
    normalization_columns = {
        instruction.target_columns[0] for instruction in normalization_instructions
    }
    before_case = before_issue_counts["case_variant_affected_cells_by_column"]
    after_case = after_issue_counts["case_variant_affected_cells_by_column"]
    targeted_case_variants_improved = all(
        after_case.get(column, 0) < before_case.get(column, 0)
        for column in normalization_columns if column in before_case
    )

    trim_columns = {
        column for instruction in executed_instructions
        if instruction.action_id == "trim_whitespace"
        for column in instruction.target_columns
    }
    targeted_whitespace_improved = all(
        after_whitespace.get(column, 0) < before_whitespace.get(column, 0)
        for column in trim_columns if column in before_whitespace
    )
    mappings = {
        instruction.target_columns[0]: instruction.parameters["mapping"]
        for instruction in normalization_instructions
    }
    only_approved_value_changes = True
    no_unexpected_categories = True
    for column in normalization_columns:
        mapping = mappings[column]
        expected = original[column].copy(deep=True)
        if column in trim_columns:
            for index, value in expected.items():
                if isinstance(value, str):
                    expected.at[index] = value.strip()
        for raw_value, canonical_value in mapping.items():
            exact_match = expected.apply(
                lambda value: isinstance(value, str) and value == raw_value
            )
            expected.loc[exact_match] = canonical_value
        if not expected.equals(cleaned[column]):
            only_approved_value_changes = False
        permitted = set(original[column].dropna().tolist()) | set(mapping.values())
        if column in trim_columns:
            permitted |= {
                value.strip() if isinstance(value, str) else value
                for value in original[column].dropna().tolist()
            }
        if not set(cleaned[column].dropna().tolist()) <= permitted:
            no_unexpected_categories = False
    left = original.astype(object).where(original.notna(), "")
    right = cleaned.astype(object).where(cleaned.notna(), "")
    equal_cells = (left.eq(right) & original.notna() & cleaned.notna()) | (original.isna() & cleaned.isna())
    for column in original.columns:
        if original[column].dtype != cleaned[column].dtype:
            equal_cells[column] = original[column].isna() & cleaned[column].isna()
    changed_cell_count = int((~equal_cells.fillna(False)).to_numpy().sum())
    newly_introduced_null_count = int((original.notna() & cleaned.isna()).to_numpy().sum())
    date_instructions = [item for item in executed_instructions if item.action_id == "parse_dates"]
    only_approved_date_changes = True
    targeted_date_formats_improved = True
    for instruction in date_instructions:
        column = instruction.target_columns[0]
        expected_input = original[[column]].copy(deep=True)
        # Replay the explicit preceding actions on this column, in execution order.
        from pipelines.pipeline1_cleaning.orchestration.action_registry import get_action
        for preceding in executed_instructions:
            if preceding.instruction_id == instruction.instruction_id:
                break
            if column in preceding.target_columns:
                get_action(preceding.action_id)(expected_input, [column], preceding.parameters)
        try:
            expected, stats = assess_dates(expected_input[column], instruction.parameters["output_format"])
            require_safe_dates(stats)
            only_approved_date_changes &= expected.equals(cleaned[column])
        except ValueError:
            only_approved_date_changes = False
        before_dates = before_issue_counts["date_format_affected_cells_by_column"].get(column, 0)
        after_dates = after_issue_counts["date_format_affected_cells_by_column"].get(column, 0)
        if before_dates:
            targeted_date_formats_improved &= after_dates < before_dates

    numeric_instructions = [item for item in executed_instructions if item.action_id == "coerce_dtype"]
    numeric_columns = {item.target_columns[0] for item in numeric_instructions}
    mask_instructions = [
        item for item in executed_instructions if item.action_id == "mask_column"
    ]
    mask_columns = {item.target_columns[0] for item in mask_instructions}
    only_approved_numeric_changes = True
    targeted_mixed_numeric_improved = True
    expected_dtype_changes_only = all(
        original[column].dtype == cleaned[column].dtype
        for column in original.columns
        if column not in numeric_columns | mask_columns
    )
    for instruction in numeric_instructions:
        column = instruction.target_columns[0]
        expected_input = original[[column]].copy(deep=True)
        from pipelines.pipeline1_cleaning.orchestration.action_registry import get_action
        try:
            for preceding in executed_instructions:
                if preceding.instruction_id == instruction.instruction_id:
                    break
                if column in preceding.target_columns:
                    get_action(preceding.action_id)(expected_input, [column], preceding.parameters)
            expected, _ = convert_numeric(expected_input[column], instruction.parameters)
            only_approved_numeric_changes &= expected.equals(cleaned[column])
            expected_dtype_changes_only &= str(cleaned[column].dtype) == TARGET_DTYPES[instruction.parameters["target_dtype"]]
        except (ValueError, KeyError):
            only_approved_numeric_changes = False
            expected_dtype_changes_only = False
        before_numeric = before_issue_counts["mixed_numeric_affected_cells_by_column"].get(column, 0)
        after_numeric = after_issue_counts["mixed_numeric_affected_cells_by_column"].get(column, 0)
        targeted_mixed_numeric_improved &= after_numeric < before_numeric if before_numeric else after_numeric == 0
    only_approved_privacy_changes = True
    for instruction in mask_instructions:
        column = instruction.target_columns[0]
        expected_input = original[[column]].copy(deep=True)
        from pipelines.pipeline1_cleaning.orchestration.action_registry import get_action
        try:
            for preceding in executed_instructions:
                if preceding.instruction_id == instruction.instruction_id:
                    break
                if column in preceding.target_columns:
                    get_action(preceding.action_id)(
                        expected_input,
                        [column],
                        preceding.parameters,
                    )
            get_action("mask_column")(
                expected_input,
                [column],
                instruction.parameters,
            )
            only_approved_privacy_changes &= expected_input[column].equals(
                cleaned[column]
            )
        except (ValueError, KeyError, TypeError):
            only_approved_privacy_changes = False
    null_positions_equal = original.isna().equals(cleaned.isna())

    comparisons = {
        "row_count_equal": before["row_count"] == after["row_count"],
        "column_count_equal": before["column_count"] == after["column_count"],
        "missingness_equal": (
            before["total_missing_cells"] == after["total_missing_cells"]
        ),
        "duplicate_row_count_equal": (
            before["duplicate_row_count"] == after["duplicate_row_count"]
        ),
        "physical_dtypes_equal": before["physical_dtypes"] == after["physical_dtypes"],
        "cardinality_equal": before["cardinality"] == after["cardinality"],
    }
    if expect_no_changes:
        no_unexpected_changes = dataframe_equal and all(comparisons.values())
    else:
        no_unexpected_changes = (
            comparisons["row_count_equal"]
            and comparisons["column_count_equal"]
            and comparisons["missingness_equal"]
            and expected_dtype_changes_only
            and unrelated_columns_equal
            and targeted_whitespace_improved
            and targeted_case_variants_improved
            and only_approved_value_changes
            and no_unexpected_categories
            and only_approved_date_changes
            and targeted_date_formats_improved
            and newly_introduced_null_count == 0
            and null_positions_equal
            and only_approved_numeric_changes
            and targeted_mixed_numeric_improved
            and only_approved_privacy_changes
        )
    warnings = [] if no_unexpected_changes else [
        "Post-clean validation found changes outside the approved action boundaries."
    ]
    report = ValidationReport(
        passed=no_unexpected_changes,
        no_unexpected_changes=no_unexpected_changes,
        dataframe_equal=dataframe_equal,
        before_summary=before,
        after_summary=after,
        unrelated_columns_equal=unrelated_columns_equal,
        targeted_whitespace_improved=targeted_whitespace_improved,
        targeted_case_variants_improved=targeted_case_variants_improved,
        only_approved_value_changes=only_approved_value_changes,
        no_unexpected_categories=no_unexpected_categories,
        targeted_date_formats_improved=targeted_date_formats_improved,
        only_approved_date_changes=only_approved_date_changes,
        expected_dtype_changes_only=expected_dtype_changes_only,
        only_approved_numeric_changes=only_approved_numeric_changes,
        only_approved_privacy_changes=only_approved_privacy_changes,
        null_positions_equal=null_positions_equal,
        targeted_mixed_numeric_improved=targeted_mixed_numeric_improved,
        newly_introduced_null_count=newly_introduced_null_count,
        changed_cell_count=changed_cell_count,
        before_issue_counts=before_issue_counts,
        after_issue_counts=after_issue_counts,
        warnings=warnings,
        **comparisons,
    )
    return report, after_profile


def validate_post_clean(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    before_profile: Dict[str, Any],
    expect_no_changes: bool = False,
    modified_columns: list[str] | None = None,
    executed_instructions: list[CleaningInstruction] | None = None,
) -> tuple[ValidationReport, Dict[str, Any]]:
    instructions = executed_instructions or []
    imputations = [item for item in instructions if item.action_id == "impute_missing"]
    if not imputations:
        return _validate_non_imputation(original, cleaned, before_profile, expect_no_changes,
                                        modified_columns, instructions)

    # Previous actions retain their existing validation against a view in which
    # approved imputation cells are still missing. Validate the fill separately.
    before_fill_view = cleaned.copy(deep=True)
    for item in imputations:
        column = item.target_columns[0]
        before_fill_view.loc[original[column].isna(), column] = original.loc[original[column].isna(), column]
    base, _ = _validate_non_imputation(
        original, before_fill_view, before_profile, expect_no_changes, modified_columns,
        [item for item in instructions if item.action_id != "impute_missing"],
    )
    from pipelines.pipeline1_cleaning.orchestration.action_registry import get_action
    from pipelines.pipeline1_cleaning.orchestration.data_cleaner_performer import _changed_cells
    audit, total_imputed, valid = [], 0, True
    seen = set()
    for item in imputations:
        column = item.target_columns[0]
        if column in seen:
            valid = False
            continue
        seen.add(column)
        expected_input = original[[column]].copy(deep=True)
        try:
            for preceding in instructions:
                if preceding.instruction_id == item.instruction_id:
                    break
                if column in preceding.target_columns:
                    get_action(preceding.action_id)(expected_input, [column], preceding.parameters)
            expected, metadata = prepare_imputation(expected_input[column], item.parameters)
            valid &= expected.equals(cleaned[column])
            observed_mask = expected_input[column].notna()
            valid &= expected_input.loc[observed_mask, column].equals(cleaned.loc[observed_mask, column])
            total_imputed += metadata["affected_cell_count"]
            audit.append({"instruction_id": item.instruction_id, "source_decision_id": item.source_decision_id,
                          "source_issue_ids": item.source_issue_ids, "column": column, **metadata})
        except (ValueError, TypeError, KeyError):
            valid = False
    after_profile = profile_dataset(cleaned)
    after = _profile_summary(after_profile)
    new_nulls = int((original.notna() & cleaned.isna()).to_numpy().sum())
    expected_missingness = (
        valid and new_nulls == 0
        and base.before_summary["total_missing_cells"] - after["total_missing_cells"] == total_imputed
    )
    passed = base.passed and valid and expected_missingness
    updates = {
        "passed": passed, "no_unexpected_changes": passed,
        "dataframe_equal": original.equals(cleaned),
        "after_summary": after, "after_issue_counts": _issue_counts(cleaned, after_profile),
        "missingness_equal": base.before_summary["total_missing_cells"] == after["total_missing_cells"],
        "null_positions_equal": original.isna().equals(cleaned.isna()),
        "physical_dtypes_equal": base.before_summary["physical_dtypes"] == after["physical_dtypes"],
        "cardinality_equal": base.before_summary["cardinality"] == after["cardinality"],
        "duplicate_row_count_equal": base.before_summary["duplicate_row_count"] == after["duplicate_row_count"],
        "newly_introduced_null_count": new_nulls,
        "only_approved_imputation_changes": valid,
        "expected_missingness_changes_only": expected_missingness,
        "imputed_cell_count": total_imputed,
        "imputation_audit": audit,
        "changed_cell_count": _changed_cells(original, cleaned)[0],
        "warnings": base.warnings + ([] if passed else ["Imputation changes did not match the approved deterministic strategy."]),
    }
    return ValidationReport.model_validate({**base.model_dump(), **updates}), after_profile
