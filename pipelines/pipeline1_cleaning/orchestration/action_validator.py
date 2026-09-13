"""Compile validated decisions into allow-listed cleaning instructions."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from pipelines.pipeline1_cleaning.decisions.decision_schema import (
    CleaningDecision,
    CleaningInstruction,
)
from pipelines.pipeline1_cleaning.orchestration.action_registry import ACTION_REGISTRY
from pipelines.pipeline1_cleaning.techniques.numeric_coercion import QUANTITATIVE_TYPES, convert_numeric
from pipelines.pipeline1_cleaning.techniques.missing_values import (
    ImputationParameters, prepare_imputation, validate_imputation_semantics,
)
from pipelines.pipeline1_cleaning.techniques.date_parsing import (
    OUTPUT_FORMATS, assess_dates, require_safe_dates,
)
from pipelines.pipeline1_cleaning.techniques.privacy_masking import (
    PII_DEFAULT_STRATEGIES,
)


def compile_instruction(
    decision: CleaningDecision,
    available_columns: list[str],
    column_dtypes: dict[str, Any] | None = None,
    dataframe: pd.DataFrame | None = None,
    column_semantics: dict[str, Any] | None = None,
) -> CleaningInstruction:
    parameters = dict(decision.parameters)
    if decision.proposed_action not in ACTION_REGISTRY:
        raise ValueError(f"Unknown action_id: {decision.proposed_action}")

    if decision.proposed_action == "trim_whitespace":
        if decision.decision_type != "action":
            raise ValueError("trim_whitespace requires decision_type='action'.")
        if decision.requires_review or decision.risk_level != "low":
            raise ValueError("trim_whitespace must be approved as a low-risk action.")
        if not decision.target_columns:
            raise ValueError("trim_whitespace requires target columns.")
        if decision.parameters:
            raise ValueError("trim_whitespace does not accept parameters.")

    if decision.proposed_action == "normalize_categorical_labels":
        if decision.decision_type != "action":
            raise ValueError("normalize_categorical_labels requires decision_type='action'.")
        if decision.requires_review or decision.risk_level != "low":
            raise ValueError("normalize_categorical_labels must be an approved low-risk action.")
        if len(decision.target_columns) != 1:
            raise ValueError("normalize_categorical_labels requires exactly one target column.")
        mapping = decision.parameters.get("mapping")
        if set(decision.parameters) != {"mapping"} or not isinstance(mapping, dict) or not mapping:
            raise ValueError("normalize_categorical_labels requires one non-empty mapping.")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()):
            raise ValueError("Categorical mapping keys and values must be strings.")
        try:
            json.dumps(mapping, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Categorical mapping must be JSON serializable.") from exc
        column = decision.target_columns[0]
        if column_dtypes is None or column not in column_dtypes:
            raise ValueError("Target column dtype is required for categorical normalization.")
        dtype = column_dtypes[column]
        if not (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        ):
            raise ValueError("Categorical normalization requires a string-compatible target column.")

    unknown_columns = sorted(set(decision.target_columns) - set(available_columns))
    if unknown_columns:
        raise ValueError(
            "Instruction targets unknown columns: " + ", ".join(unknown_columns)
        )

    if decision.proposed_action == "parse_dates":
        if decision.decision_type != "action" or decision.requires_review or decision.risk_level != "low":
            raise ValueError("parse_dates requires an approved low-risk action without review.")
        if len(decision.target_columns) != 1:
            raise ValueError("parse_dates requires exactly one target column.")
        column = decision.target_columns[0]
        semantic = (column_semantics if column_semantics is not None else decision.semantic_basis).get(column, {})
        kind = semantic.get("semantic_type")
        if kind not in OUTPUT_FORMATS or semantic.get("value_function") != "temporal":
            raise ValueError("parse_dates requires date/datetime/time semantics and temporal value_function.")
        if set(parameters) - {"output_format"}:
            raise ValueError("parse_dates accepts only output_format.")
        parameters.setdefault("output_format", OUTPUT_FORMATS[kind])
        if parameters["output_format"] != OUTPUT_FORMATS[kind]:
            raise ValueError("parse_dates output_format must match the semantic type allow-list.")
        if dataframe is None or column not in dataframe.columns:
            raise ValueError("parse_dates requires observed column values for coverage checks.")
        if not (pd.api.types.is_object_dtype(dataframe[column].dtype) or pd.api.types.is_string_dtype(dataframe[column].dtype)):
            raise ValueError("parse_dates currently supports string-compatible columns only.")
        _, stats = assess_dates(dataframe[column], parameters["output_format"])
        require_safe_dates(stats)

    if decision.proposed_action == "no_op":
        if decision.target_columns:
            raise ValueError("no_op must not target individual columns.")
        if decision.parameters:
            raise ValueError("no_op does not accept parameters.")

    if decision.proposed_action == "mask_column":
        if (
            decision.decision_type != "action"
            or decision.decision_source != "deterministic_tier1"
            or decision.requires_review
            or decision.risk_level != "low"
        ):
            raise ValueError(
                "mask_column requires an approved deterministic Tier 1 "
                "low-risk action without review."
            )
        if len(decision.target_columns) != 1:
            raise ValueError("mask_column requires exactly one target column.")
        subtype = decision.evidence_summary.get("pii_subtype")
        expected_strategy = PII_DEFAULT_STRATEGIES.get(subtype)
        if expected_strategy is None:
            raise ValueError("mask_column requires a recognized PII subtype.")
        if parameters != {"strategy": expected_strategy}:
            raise ValueError(
                "mask_column strategy must match the deterministic subtype default."
            )
        if decision.evidence_summary.get("privacy_reason") != "masked for privacy":
            raise ValueError("mask_column requires the privacy audit reason.")
        column = decision.target_columns[0]
        if dataframe is None or column not in dataframe.columns:
            raise ValueError("mask_column requires observed target values.")
        if not (
            pd.api.types.is_object_dtype(dataframe[column].dtype)
            or pd.api.types.is_string_dtype(dataframe[column].dtype)
        ):
            raise ValueError("mask_column currently requires string-compatible values.")

    if decision.proposed_action == "coerce_dtype":
        if decision.decision_type != "action" or decision.requires_review or decision.risk_level != "low":
            raise ValueError("coerce_dtype requires an approved low-risk action without review.")
        if len(decision.target_columns) != 1:
            raise ValueError("coerce_dtype requires exactly one target column.")
        column = decision.target_columns[0]
        semantic = (column_semantics if column_semantics is not None else decision.semantic_basis).get(column, {})
        if semantic.get("semantic_type") not in QUANTITATIVE_TYPES or semantic.get("value_function") != "measurement":
            raise ValueError("coerce_dtype requires quantitative semantics and measurement value_function.")
        if dataframe is None or column not in dataframe.columns:
            raise ValueError("coerce_dtype requires observed values for numeric coverage checks.")
        _, stats = convert_numeric(dataframe[column], parameters)
        parameters = {"target_dtype": stats["chosen_target_dtype"]}

    if decision.proposed_action == "impute_missing":
        if decision.decision_type != "action" or decision.requires_review or decision.risk_level != "low":
            raise ValueError("impute_missing requires an explicitly approved low-risk action without review.")
        if len(decision.target_columns) != 1:
            raise ValueError("impute_missing requires exactly one target column.")
        column = decision.target_columns[0]
        semantic = (column_semantics if column_semantics is not None else decision.semantic_basis).get(column, {})
        policy = ImputationParameters.model_validate(parameters)
        validate_imputation_semantics(semantic, decision.evidence_summary, policy)
        if dataframe is None or column not in dataframe.columns:
            raise ValueError("Imputation requires observed column values and missingness evidence.")
        prepare_imputation(dataframe[column], parameters)
        parameters = policy.model_dump(exclude_none=True)

    return CleaningInstruction(
        instruction_id=f"instruction-{decision.decision_id}",
        source_decision_id=decision.decision_id,
        source_issue_ids=list(decision.source_issue_ids),
        action_id=decision.proposed_action,
        target_columns=list(decision.target_columns),
        parameters=parameters,
    )
