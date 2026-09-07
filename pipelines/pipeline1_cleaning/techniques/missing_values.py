"""Conservative deterministic imputation; replacement values are never model inputs."""

import math
from decimal import Decimal
from numbers import Real
from typing import Any, Dict, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from pipelines.pipeline1_cleaning.techniques.numeric_coercion import QUANTITATIVE_TYPES


MIN_OBSERVED_VALUES = 20
MIN_SEMANTIC_CONFIDENCE = 0.8
NUMERIC_LABEL_RISK_MIN_INTEGRAL_FRACTION = 0.99
NUMERIC_LABEL_RISK_MIN_UNIQUE_RATIO = 0.50
NUMERIC_LABEL_RISK_MAX_REPETITION_RATE = 0.50
CONSTANTS = {
    "unknown": "Unknown", "not_available": "Not Available",
    "not_applicable": "Not Applicable", "true": True, "false": False,
}


def _median_semantic_result(status, reason_code, reason, evidence):
    return {
        "semantic_eligible": status,
        "reason_code": reason_code,
        "reason": reason,
        "evidence": evidence,
    }


def assess_median_semantic_eligibility(
    semantic: Dict[str, Any], series: pd.Series,
    issue_evidence: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assess meaning, not mechanical feasibility, for median imputation.

    The numeric-shape heuristic is a veto only. It can turn an otherwise positive
    semantic claim into unknown, but can never establish measurement meaning.
    """
    issue_evidence = issue_evidence or {}
    observed = series[series.notna()]
    observed_count = len(observed)
    unique_ratio = float(observed.nunique(dropna=True) / observed_count) if observed_count else 0.0
    counts = observed.value_counts(dropna=True)
    repeated_occurrences = int(counts[counts > 1].sum())
    repetition_rate = float(repeated_occurrences / observed_count) if observed_count else 0.0
    numeric = pd.to_numeric(observed, errors="coerce")
    numeric_values = np.asarray(numeric, dtype=float) if observed_count else np.asarray([], dtype=float)
    finite = np.isfinite(numeric_values)
    integral = finite & np.isclose(numeric_values, np.rint(numeric_values), rtol=0.0, atol=1e-12)
    integral_fraction = float(integral.sum() / observed_count) if observed_count else 0.0

    kind = semantic.get("semantic_type")
    function = semantic.get("value_function")
    numeric_meaning = semantic.get("numeric_meaning")
    role = semantic.get("semantic_role")
    confidence = semantic.get("confidence")
    evidence = {
        "semantic_type": kind,
        "value_function": function,
        "numeric_meaning": numeric_meaning,
        "semantic_role": role,
        "semantic_confidence": confidence,
        "concept_description": semantic.get("concept_description"),
        "physical_dtype": str(series.dtype),
        "observed_count": observed_count,
        "integral_fraction": integral_fraction,
        "unique_ratio": unique_ratio,
        "repetition_rate": repetition_rate,
        "identifier_like": issue_evidence.get("identifier_like") is True,
    }

    identifier_roles = {"primary_identifier", "foreign_identifier", "transaction_identifier"}
    if kind == "identifier" or function == "entity_reference" or role in identifier_roles or evidence["identifier_like"]:
        return _median_semantic_result(
            "false", "identifier_or_entity_reference",
            "Identifier or entity-reference meaning blocks median imputation.", evidence,
        )
    blocked_meanings = {
        "ordinal": "ordered_values_are_not_magnitudes",
        "nominal_code": "nominal_numeric_code",
        "identifier": "numeric_identifier",
        "not_applicable": "numeric_meaning_not_applicable",
    }
    if numeric_meaning in blocked_meanings:
        return _median_semantic_result(
            "false", blocked_meanings[numeric_meaning],
            "The declared numeric meaning does not support median imputation.", evidence,
        )
    if numeric_meaning in {None, "unknown"}:
        return _median_semantic_result(
            "unknown", "numeric_meaning_unknown",
            "Numeric meaning is missing or unknown.", evidence,
        )
    if (
        not isinstance(confidence, Real) or isinstance(confidence, bool)
        or not math.isfinite(float(confidence)) or float(confidence) < MIN_SEMANTIC_CONFIDENCE
    ):
        return _median_semantic_result(
            "unknown", "insufficient_semantic_confidence",
            "Semantic confidence is below the required 0.8 threshold.", evidence,
        )
    if numeric_meaning != "magnitude" or kind not in QUANTITATIVE_TYPES or function != "measurement":
        return _median_semantic_result(
            "unknown", "contradictory_quantitative_semantics",
            "Numeric meaning, semantic type, and value function are inconsistent.", evidence,
        )
    if (
        integral_fraction >= NUMERIC_LABEL_RISK_MIN_INTEGRAL_FRACTION
        and unique_ratio >= NUMERIC_LABEL_RISK_MIN_UNIQUE_RATIO
        and repetition_rate < NUMERIC_LABEL_RISK_MAX_REPETITION_RATE
    ):
        return _median_semantic_result(
            "unknown", "high_cardinality_integral_label_risk",
            "Highly unique, weakly repeated integral values may be numeric labels; the heuristic only vetoes authorization.",
            evidence,
        )
    return _median_semantic_result(
        "true", "confirmed_quantitative_magnitude",
        "The semantic contract identifies a quantitative magnitude and no numeric-label risk veto applies.",
        evidence,
    )


class ImputationParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["median", "mode", "constant"]
    constant_type: Literal["unknown", "not_available", "not_applicable", "true", "false"] | None = None

    @model_validator(mode="after")
    def check_constant(self):
        if (self.strategy == "constant") != (self.constant_type is not None):
            raise ValueError("constant_type is required only for constant imputation.")
        return self


def missingness_assessment(series):
    count = int(series.isna().sum())
    percentage = 100 * count / len(series) if len(series) else 0.0
    risk = ("none" if not count else "low" if percentage <= 5 else
            "moderate" if percentage <= 20 else "high" if percentage <= 50 else "very_high")
    return {"missing_count": count, "missing_percentage": percentage,
            "missingness_risk": risk, "observed_count": len(series) - count}


def validate_imputation_semantics(semantic, evidence, parameters):
    kind, function = semantic.get("semantic_type"), semantic.get("value_function")
    if semantic.get("semantic_role") in {"primary_identifier", "foreign_identifier", "transaction_identifier"} or evidence.get("identifier_like") is True:
        raise ValueError("Identifier/reference evidence blocks automatic imputation.")
    confidence = semantic.get("confidence")
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or confidence < MIN_SEMANTIC_CONFIDENCE:
        raise ValueError("Uncertain semantic meaning blocks imputation (confidence must be >=0.8).")
    if evidence.get("missingness_is_meaningful") is True:
        raise ValueError("Known meaningful missingness must be preserved.")
    if kind in QUANTITATIVE_TYPES and function == "measurement":
        if parameters.strategy != "median":
            raise ValueError("Quantitative measurements permit only median imputation.")
        return
    if kind == "categorical" and function == "classification":
        if parameters.strategy == "mode":
            if evidence.get("missingness_is_meaningful") is not False:
                raise ValueError("Mode requires explicit evidence that missingness is not meaningful.")
        elif parameters.strategy == "constant" and parameters.constant_type in {"unknown", "not_available", "not_applicable"}:
            if evidence.get("approved_missingness_interpretation") != parameters.constant_type:
                raise ValueError("Constant requires an explicitly approved matching missingness interpretation.")
        else:
            raise ValueError("Categorical imputation permits mode or a controlled neutral constant only.")
    elif kind == "boolean" and function == "boolean_state":
        if parameters.strategy != "constant" or parameters.constant_type not in {"true", "false"}:
            raise ValueError("Boolean NULL must be preserved unless a justified constant state is approved.")
        if evidence.get("approved_missingness_interpretation") != parameters.constant_type:
            raise ValueError("Boolean constant requires explicit evidence that NULL logically means that state.")
    else:
        raise ValueError("Semantic class is blocked from automatic V1 imputation.")
    justification = evidence.get("missingness_justification")
    if not isinstance(justification, str) or not justification.strip():
        raise ValueError("An explicit semantic justification for the missingness strategy is required.")


def prepare_imputation(series, parameters):
    policy = ImputationParameters.model_validate(parameters)
    audit = missingness_assessment(series)
    if audit["missingness_risk"] != "low":
        raise ValueError(f"Imputation requires LOW missingness (0–5%); observed {audit['missingness_risk']}.")
    observed = series[series.notna()]
    if policy.strategy in {"median", "mode"} and len(observed) < MIN_OBSERVED_VALUES:
        raise ValueError("Median/mode requires at least 20 observed values.")
    if policy.strategy == "median":
        if pd.api.types.is_bool_dtype(series.dtype) or not (
            pd.api.types.is_integer_dtype(series.dtype) or pd.api.types.is_float_dtype(series.dtype)
        ):
            raise ValueError("Median requires an already numeric physical dtype; identifier-like text is not coerced.")
        numbers = sorted(Decimal(str(value)) for value in observed.tolist())
        if any(not value.is_finite() for value in numbers):
            raise ValueError("Median requires finite numeric observations.")
        middle = len(numbers) // 2
        median = numbers[middle] if len(numbers) % 2 else (numbers[middle - 1] + numbers[middle]) / 2
        if pd.api.types.is_integer_dtype(series.dtype):
            if median != median.to_integral_value():
                raise ValueError("Non-integer median cannot be stored without changing the integer representation; review required.")
            replacement = int(median)
        else:
            replacement = float(median)
            if not math.isfinite(replacement) or Decimal(str(replacement)) != median:
                raise ValueError("Median cannot be represented safely in the target dtype.")
    elif policy.strategy == "mode":
        if not all(isinstance(value, str) for value in observed):
            raise ValueError("Mode requires observed categorical string labels; boolean mode is not supported.")
        counts = observed.value_counts()
        if int((counts == counts.max()).sum()) != 1:
            raise ValueError("Tied modes are ambiguous; missing values preserved.")
        replacement = counts.index[0]
    else:
        replacement = CONSTANTS[policy.constant_type]
        if isinstance(replacement, bool):
            if not all(isinstance(value, (bool, np.bool_)) for value in observed):
                raise ValueError("Boolean constants require observed boolean values.")
        elif not all(isinstance(value, str) for value in observed):
            raise ValueError("Neutral constants require observed categorical string labels.")
    filled = series.copy(deep=True)
    mask = series.isna()
    try:
        filled.loc[mask] = replacement
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Replacement is incompatible with the existing physical dtype.") from exc
    if filled.dtype != series.dtype or not filled.loc[~mask].equals(observed):
        raise ValueError("Imputation must preserve dtype and every observed value.")
    if filled.isna().any() or not all(value == replacement for value in filled.loc[mask]):
        raise ValueError("Replacement cannot be stored exactly; missing values preserved.")
    audit.update({"strategy": policy.strategy, "constant_type": policy.constant_type,
                  "replacement_value": replacement, "affected_cell_count": audit["missing_count"],
                  "dtype_before": str(series.dtype), "dtype_after": str(filled.dtype)})
    return filled, audit


def impute_missing(dataframe, target_columns, parameters):
    if len(target_columns) != 1:
        raise ValueError("impute_missing requires exactly one target column.")
    column = target_columns[0]
    filled, _ = prepare_imputation(dataframe[column], parameters)
    dataframe[column] = filled
    return dataframe
