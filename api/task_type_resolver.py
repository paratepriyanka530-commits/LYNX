"""Deterministic target-to-task resolution for the LYNX application.

The resolver is intentionally model-independent and conservative.  It combines
the observed shape of the *cleaned* target with the effective Pipeline 1
semantic record when one is available.  Ambiguous or unsuitable targets are
reported as unresolved instead of being guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd


ResolutionStatus = Literal["resolved", "unresolved"]
TaskType = Literal["classification", "regression"]

MIN_USABLE_OBSERVATIONS = 4
STRONG_SEMANTIC_CONFIDENCE = 0.75

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})"
    r"(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)
_IDENTIFIER_SEMANTIC_TYPES = {"identifier", "contact_information"}
_IDENTIFIER_ROLES = {
    "primary_identifier", "foreign_identifier", "transaction_identifier",
}
_TEMPORAL_TYPES = {"date", "time", "datetime"}


@dataclass(frozen=True)
class TaskTypeResolution:
    """Auditable result returned by :func:`resolve_task_type`."""

    status: ResolutionStatus
    task_type: TaskType | None
    reason: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "task_type": self.task_type,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        return dumped if isinstance(dumped, Mapping) else {}
    return {}


def _semantic_evidence(
    pipeline1_report: Any, target_column: str,
) -> dict[str, Any]:
    """Extract the selected column's effective Architecture-B semantic record."""
    report = _mapping(pipeline1_report)
    profile = _mapping(report.get("semantic_profile"))
    matches: list[Mapping[str, Any]] = []
    for item in profile.get("columns", []) or []:
        candidate = _mapping(item)
        if candidate.get("column_name") == target_column:
            matches.append(candidate)

    result: dict[str, Any] = {
        "semantic_available": len(matches) == 1,
        "semantic_match_count": len(matches),
    }
    if len(matches) == 1:
        semantic = matches[0]
        for source, destination in (
            ("semantic_type", "semantic_type"),
            ("semantic_role", "semantic_role"),
            ("value_function", "value_function"),
            ("numeric_meaning", "numeric_meaning"),
            ("concept_description", "semantic_concept"),
            ("reasoning_summary", "semantic_reason"),
        ):
            if semantic.get(source) is not None:
                result[destination] = semantic.get(source)
        confidence = semantic.get("confidence")
        if confidence is not None:
            try:
                result["semantic_confidence"] = float(confidence)
            except (TypeError, ValueError):
                result["semantic_confidence"] = None

    guard = _mapping(_mapping(profile.get("telemetry")).get("semantic_contradiction_guard"))
    guard_entries = []
    for item in guard.get("entries", []) or []:
        entry = _mapping(item)
        if entry.get("column_name") == target_column:
            guard_entries.append(entry)
    if guard_entries:
        entry = guard_entries[0]
        result["semantic_contradiction_detected"] = bool(
            entry.get("contradiction_detected")
        )
        result["semantic_contradiction_code"] = entry.get("contradiction_code")
    return result


def _identifier_name_signal(column_name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", column_name.casefold()).strip("_")
    tokens = [token for token in normalized.split("_") if token]
    if normalized in {"id", "uuid", "guid"}:
        return True
    if tokens and tokens[-1] in {"id", "uuid", "guid", "identifier"}:
        return True
    # Cover common compact/camel-collapsed forms such as customerID.
    return bool(re.search(r"(?:customer|account|patient|employee|student|user|order|transaction)id$", normalized))


def _string_shape(non_null: pd.Series) -> dict[str, Any]:
    values = non_null.astype(str).str.strip()
    lengths = values.str.len()
    word_counts = values.str.split().str.len()
    uuid_coverage = float(values.str.fullmatch(_UUID_RE).mean()) if len(values) else 0.0
    date_coverage = float(values.str.fullmatch(_DATE_RE).mean()) if len(values) else 0.0
    return {
        "maximum_text_length": int(lengths.max()) if len(lengths) else 0,
        "median_text_length": float(lengths.median()) if len(lengths) else 0.0,
        "mean_word_count": float(word_counts.mean()) if len(word_counts) else 0.0,
        "uuid_pattern_coverage": round(uuid_coverage, 6),
        "date_pattern_coverage": round(date_coverage, 6),
    }


def _target_evidence(series: pd.Series, target_column: str) -> dict[str, Any]:
    non_null = series.dropna()
    non_null_count = int(len(non_null))
    unique_count = int(non_null.nunique(dropna=True)) if non_null_count else 0
    unique_ratio = unique_count / non_null_count if non_null_count else 0.0
    is_bool_dtype = bool(pd.api.types.is_bool_dtype(series.dtype))
    object_boolean = bool(
        non_null_count
        and all(isinstance(value, (bool, np.bool_)) for value in non_null.tolist())
    )
    is_boolean = is_bool_dtype or object_boolean
    is_numeric = bool(pd.api.types.is_numeric_dtype(series.dtype) and not is_boolean)
    is_datetime = bool(pd.api.types.is_datetime64_any_dtype(series.dtype))

    evidence: dict[str, Any] = {
        "dtype": str(series.dtype),
        "row_count": int(len(series)),
        "missing_count": int(series.isna().sum()),
        "non_null_count": non_null_count,
        "unique_count": unique_count,
        "unique_ratio": round(unique_ratio, 6),
        "is_boolean": is_boolean,
        "is_numeric": is_numeric,
        "is_datetime": is_datetime,
        "is_integer_like": False,
        "identifier_name_signal": _identifier_name_signal(target_column),
    }

    counts = non_null.value_counts(dropna=True) if non_null_count else pd.Series(dtype=int)
    evidence["maximum_value_frequency"] = int(counts.max()) if len(counts) else 0
    evidence["minimum_value_frequency"] = int(counts.min()) if len(counts) else 0
    evidence["repeated_observation_count"] = int(counts[counts > 1].sum()) if len(counts) else 0

    if is_numeric and non_null_count:
        numeric = pd.to_numeric(non_null, errors="coerce").to_numpy(dtype=float)
        finite = numeric[np.isfinite(numeric)]
        evidence["finite_numeric_count"] = int(len(finite))
        evidence["non_finite_numeric_count"] = int(len(numeric) - len(finite))
        if len(finite):
            integer_like = bool(np.all(np.isclose(finite, np.round(finite))))
            evidence["is_integer_like"] = integer_like
            evidence["numeric_min"] = float(np.min(finite))
            evidence["numeric_max"] = float(np.max(finite))
            evidence["numeric_std"] = float(np.std(finite))
            evidence["fractional_value_ratio"] = round(
                float(np.mean(~np.isclose(finite, np.round(finite)))), 6
            )
    elif non_null_count and (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        evidence.update(_string_shape(non_null))
    return evidence


def _unresolved(reason: str, evidence: dict[str, Any]) -> TaskTypeResolution:
    return TaskTypeResolution("unresolved", None, reason, evidence)


def _resolved(
    task_type: TaskType, reason: str, evidence: dict[str, Any],
) -> TaskTypeResolution:
    return TaskTypeResolution("resolved", task_type, reason, evidence)


def resolve_task_type(
    dataframe: pd.DataFrame,
    target_column: str,
    pipeline1_report: Any | None = None,
) -> TaskTypeResolution:
    """Resolve a cleaned target to classification, regression, or unresolved.

    Benchmark labels and model-specific heuristics are deliberately not inputs.
    Missing target values are excluded from evidence and are never imputed.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if not isinstance(target_column, str) or not target_column:
        raise ValueError("target_column must be a non-empty string")

    occurrences = list(dataframe.columns).count(target_column)
    if occurrences == 0:
        return _unresolved(
            f"Target column {target_column!r} does not exist in the prepared dataset.",
            {"target_column": target_column, "target_column_occurrences": 0},
        )
    if occurrences != 1:
        return _unresolved(
            f"Target column {target_column!r} is ambiguous because it occurs {occurrences} times.",
            {"target_column": target_column, "target_column_occurrences": occurrences},
        )

    series = dataframe[target_column]
    evidence = {
        "target_column": target_column,
        "target_column_occurrences": 1,
        **_target_evidence(series, target_column),
        **_semantic_evidence(pipeline1_report, target_column),
    }

    non_null_count = int(evidence["non_null_count"])
    unique_count = int(evidence["unique_count"])
    unique_ratio = float(evidence["unique_ratio"])
    if non_null_count == 0:
        return _unresolved("Target is all null; no usable target observations remain.", evidence)
    if non_null_count < MIN_USABLE_OBSERVATIONS:
        return _unresolved(
            f"Target has only {non_null_count} non-null observations; at least "
            f"{MIN_USABLE_OBSERVATIONS} are required for safe task resolution.",
            evidence,
        )
    if unique_count < 2:
        return _unresolved("Target is constant after null values are excluded.", evidence)

    if evidence.get("semantic_match_count", 0) > 1:
        return _unresolved(
            "Pipeline 1 contains multiple semantic records for the selected target.", evidence
        )

    semantic_type = evidence.get("semantic_type")
    semantic_role = evidence.get("semantic_role")
    value_function = evidence.get("value_function")
    numeric_meaning = evidence.get("numeric_meaning")
    confidence = evidence.get("semantic_confidence")
    strong_semantics = confidence is not None and confidence >= STRONG_SEMANTIC_CONFIDENCE

    identifier_semantics = strong_semantics and (
        semantic_type in _IDENTIFIER_SEMANTIC_TYPES
        or semantic_role in _IDENTIFIER_ROLES
        or value_function in {"entity_reference", "contact"}
        # numeric_meaning describes how numeric values function.  Older
        # Pipeline 1 artifacts occasionally populated it for string labels;
        # treating that inapplicable field as decisive would discard stronger
        # categorical value evidence.
        or (evidence["is_numeric"] and numeric_meaning == "identifier")
    )
    temporal_semantics = strong_semantics and (
        semantic_type in _TEMPORAL_TYPES or value_function == "temporal"
    )
    free_text_semantics = strong_semantics and (
        semantic_type == "text" or value_function == "descriptive_attribute"
    )
    classification_semantics = strong_semantics and (
        semantic_type in {"categorical", "boolean"}
        or semantic_role == "category"
        or value_function in {"classification", "boolean_state"}
        or numeric_meaning in {"ordinal", "nominal_code"}
    )
    regression_semantics = strong_semantics and (
        semantic_type in {"numeric_measurement", "currency", "percentage"}
        or semantic_role == "measurement"
        or value_function == "measurement"
        or numeric_meaning == "magnitude"
    )
    evidence.update({
        "strong_semantics_available": bool(strong_semantics),
        "identifier_semantic_signal": bool(identifier_semantics),
        "temporal_semantic_signal": bool(temporal_semantics),
        "free_text_semantic_signal": bool(free_text_semantics),
        "classification_semantic_signal": bool(classification_semantics),
        "regression_semantic_signal": bool(regression_semantics),
    })

    if identifier_semantics and (classification_semantics or regression_semantics):
        return _unresolved(
            "Pipeline 1 supplies conflicting strong identifier and predictive-target semantics.",
            evidence,
        )
    if identifier_semantics or evidence["identifier_name_signal"]:
        return _unresolved(
            "Target is identifier-like and is unsuitable as a classification or regression outcome.",
            evidence,
        )
    if evidence["is_datetime"] or temporal_semantics:
        return _unresolved(
            "Date/time targets have no supported prediction contract in the current model library.",
            evidence,
        )
    if evidence.get("uuid_pattern_coverage", 0.0) >= 0.95:
        return _unresolved("Target values follow a strong UUID identifier pattern.", evidence)
    if evidence.get("date_pattern_coverage", 0.0) >= 0.95:
        return _unresolved(
            "Target values follow a strong date/time pattern and are not treated as regression.",
            evidence,
        )
    if free_text_semantics:
        return _unresolved(
            "Pipeline 1 identifies the target as free-form/descriptive text.", evidence
        )
    if classification_semantics and regression_semantics:
        return _unresolved(
            "Pipeline 1 supplies conflicting strong classification and measurement semantics.",
            evidence,
        )

    if evidence["is_boolean"]:
        if regression_semantics:
            return _unresolved(
                "Observed boolean values conflict with strong measurement semantics.", evidence
            )
        return _resolved(
            "classification",
            "Resolved as classification because the non-null target values are boolean.",
            evidence,
        )

    is_numeric = bool(evidence["is_numeric"])
    repeated = int(evidence["repeated_observation_count"])
    discrete_ceiling = max(4, min(20, int(math.sqrt(non_null_count)) + 1))
    discrete_numeric_shape = bool(
        is_numeric
        and evidence["is_integer_like"]
        and 2 <= unique_count <= discrete_ceiling
        and repeated > unique_count
        and unique_ratio <= 0.5
    )
    continuous_numeric_shape = bool(
        is_numeric
        and evidence.get("non_finite_numeric_count", 0) == 0
        and evidence.get("numeric_std", 0.0) > 0.0
        and unique_count >= max(5, int(math.sqrt(non_null_count)))
        and unique_ratio >= 0.5
    )
    fractional_continuous_shape = bool(
        continuous_numeric_shape and evidence.get("fractional_value_ratio", 0.0) >= 0.2
    )
    evidence.update({
        "discrete_numeric_shape": discrete_numeric_shape,
        "continuous_numeric_shape": continuous_numeric_shape,
        "fractional_continuous_shape": fractional_continuous_shape,
    })

    if is_numeric:
        if evidence.get("non_finite_numeric_count", 0) > 0:
            return _unresolved(
                "Numeric target contains infinite values and is not safe for supervised task resolution.",
                evidence,
            )
        if regression_semantics and discrete_numeric_shape:
            return _unresolved(
                "Repeated low-cardinality integer values conflict with strong continuous-measurement semantics.",
                evidence,
            )
        if classification_semantics and fractional_continuous_shape:
            return _unresolved(
                "Highly varied fractional values conflict with strong categorical semantics.", evidence
            )
        if classification_semantics and discrete_numeric_shape:
            return _resolved(
                "classification",
                "Resolved as classification because repeated discrete numeric values are corroborated by strong categorical/classification semantics.",
                evidence,
            )
        if regression_semantics and unique_count >= 5 and evidence.get("numeric_std", 0.0) > 0:
            return _resolved(
                "regression",
                "Resolved as regression because varied numeric values are corroborated by strong measurement/magnitude semantics.",
                evidence,
            )
        if fractional_continuous_shape:
            return _resolved(
                "regression",
                "Resolved as regression because the target has broad, finite, substantially fractional continuous numeric variation with no identifier or categorical signal.",
                evidence,
            )
        if discrete_numeric_shape:
            return _unresolved(
                "Low-cardinality integer values admit both categorical-code and ordinal/measurement interpretations, and semantics do not disambiguate them.",
                evidence,
            )
        return _unresolved(
            "Numeric evidence is insufficient to distinguish a measurement from a code or identifier safely.",
            evidence,
        )

    if regression_semantics:
        return _unresolved(
            "Strong measurement semantics conflict with a non-numeric physical target representation.",
            evidence,
        )

    maximum_length = int(evidence.get("maximum_text_length", 0))
    median_length = float(evidence.get("median_text_length", 0.0))
    mean_words = float(evidence.get("mean_word_count", 0.0))
    compact_labels = bool(maximum_length <= 64 and median_length <= 32 and mean_words <= 4)
    repeated_label_shape = bool(
        compact_labels
        and 2 <= unique_count <= discrete_ceiling
        and repeated > unique_count
        and unique_ratio <= 0.5
    )
    evidence.update({
        "compact_text_values": compact_labels,
        "repeated_label_shape": repeated_label_shape,
    })
    if classification_semantics and compact_labels and repeated > 0:
        return _resolved(
            "classification",
            "Resolved as classification because compact repeated labels are corroborated by strong categorical/classification semantics.",
            evidence,
        )
    if repeated_label_shape:
        return _resolved(
            "classification",
            "Resolved as classification because the target is a compact, repeated, finite set of non-numeric labels.",
            evidence,
        )
    return _unresolved(
        "Non-numeric target values are too varied or text-like for safe classification.",
        evidence,
    )


__all__ = ["TaskTypeResolution", "resolve_task_type"]
