"""Narrow deterministic contradiction guard for Architecture-B semantics.

This module does not infer semantics from ambiguous shapes.  It changes an LLM
result only when full-column values provide a strong, schema-representable
contradiction.  The untouched LLM result and the effective result are retained
in telemetry for auditability.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import pandas as pd

from pipelines.pipeline1_cleaning.semantics.batched_schema import (
    BatchedColumnSemantic,
    SemanticProfile,
)
from pipelines.pipeline1_cleaning.techniques.date_parsing import (
    MIN_PARSE_COVERAGE,
    OUTPUT_FORMATS,
    assess_dates,
)


GUARD_VERSION = "semantic_contradiction_guard_v001"
CORRECTION_CONFIDENCE_CAP = 0.75
MIN_PATTERN_VALUES = 3
MIN_ENUMERATED_VALUES = 20
MAX_ENUMERATED_LABELS = 20
MAX_ENUMERATED_UNIQUE_RATIO = 0.05
MIN_STRUCTURED_PATTERN_COVERAGE = 0.95


def _non_null_values(series: pd.Series) -> list[Any]:
    return [value for value in series.tolist() if not pd.isna(value)]


def _repeated_enumerated_string_labels(series: pd.Series) -> dict[str, Any] | None:
    values = _non_null_values(series)
    if len(values) < MIN_ENUMERATED_VALUES or not all(isinstance(value, str) for value in values):
        return None
    normalized = [value.strip() for value in values]
    if any(not value or len(value) > 64 for value in normalized):
        return None
    counts = pd.Series(normalized, dtype="object").value_counts(dropna=False)
    unique_count = int(len(counts))
    unique_ratio = unique_count / len(normalized)
    if not (
        2 <= unique_count <= MAX_ENUMERATED_LABELS
        and unique_ratio <= MAX_ENUMERATED_UNIQUE_RATIO
        and int(counts.min()) >= 2
    ):
        return None
    return {
        "pattern": "repeated_finite_enumerated_string_labels",
        "non_null_count": len(normalized),
        "unique_count": unique_count,
        "unique_ratio": round(unique_ratio, 6),
        "minimum_label_frequency": int(counts.min()),
    }


def _pattern_coverage(values: list[Any], predicate: Callable[[str], bool]) -> float:
    if not values or not all(isinstance(value, str) for value in values):
        return 0.0
    return sum(predicate(value.strip()) for value in values) / len(values)


def _structured_string_contradiction(series: pd.Series) -> tuple[str, dict[str, Any]] | None:
    values = _non_null_values(series)
    if len(values) < MIN_PATTERN_VALUES:
        return None

    patterns: list[tuple[str, str, Callable[[str], bool]]] = [
        (
            "uuid",
            "identifier",
            lambda value: bool(re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                value,
                re.IGNORECASE,
            )),
        ),
        (
            "email",
            "contact_information",
            lambda value: bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value)),
        ),
        (
            "url",
            "unknown",
            lambda value: bool(re.fullmatch(r"https?://\S+", value, re.IGNORECASE)),
        ),
    ]
    for pattern_name, corrected_type, predicate in patterns:
        coverage = _pattern_coverage(values, predicate)
        if coverage >= MIN_STRUCTURED_PATTERN_COVERAGE:
            return corrected_type, {
                "pattern": f"strong_{pattern_name}_format",
                "non_null_count": len(values),
                "match_coverage": round(coverage, 6),
                "minimum_coverage": MIN_STRUCTURED_PATTERN_COVERAGE,
            }

    # Reuse the production date parser's explicit-format and locale-ambiguity
    # protections instead of implementing another date recognizer.
    for semantic_type in ("datetime", "date"):
        _, stats = assess_dates(series, OUTPUT_FORMATS[semantic_type])
        if (
            stats["non_null_count"] >= MIN_PATTERN_VALUES
            and stats["parse_coverage"] >= MIN_PARSE_COVERAGE
            and stats["ambiguous_count"] == 0
        ):
            return semantic_type, {
                "pattern": f"strong_{semantic_type}_format",
                **stats,
            }
    return None


def _corrected_result(
    original: BatchedColumnSemantic,
    semantic_type: str,
    contradiction_code: str,
) -> BatchedColumnSemantic:
    mappings = {
        "categorical": ("classification", "not_applicable"),
        "date": ("temporal", "not_applicable"),
        "datetime": ("temporal", "not_applicable"),
        "identifier": ("entity_reference", "not_applicable"),
        "contact_information": ("contact", "not_applicable"),
        "unknown": ("unknown", "unknown"),
    }
    value_function, numeric_meaning = mappings[semantic_type]
    payload = original.model_dump(mode="python")
    payload.update({
        "semantic_type": semantic_type,
        "value_function": value_function,
        "numeric_meaning": numeric_meaning,
        "concept_description": (
            "Effective semantic result selected by the deterministic contradiction guard; "
            "the original LLM description is retained in semantic_contradiction_guard telemetry."
        ),
        "semantic_role": "unknown",
        "confidence": min(float(original.confidence), CORRECTION_CONFIDENCE_CAP),
        "reasoning_summary": (
            f"Deterministic contradiction guard {contradiction_code} replaced the incompatible "
            "LLM assignment using full-column value evidence."
        ),
    })
    return BatchedColumnSemantic.model_validate(payload)


def apply_semantic_contradiction_guard(
    profile: SemanticProfile,
    dataframe: pd.DataFrame,
) -> SemanticProfile:
    """Return effective semantics plus a raw/effective audit trail.

    Binary and low-cardinality numeric shapes are intentionally absent from
    these rules and therefore can never cause an override.
    """
    effective_columns: list[BatchedColumnSemantic] = []
    audit_entries: list[dict[str, Any]] = []
    correction_count = 0

    for original in profile.columns:
        effective = original
        contradiction_code = None
        evidence = None
        if original.column_name in dataframe.columns:
            series = dataframe[original.column_name]
            if original.semantic_type == "text":
                evidence = _repeated_enumerated_string_labels(series)
                if evidence is not None:
                    contradiction_code = "enumerated_labels_contradict_text"
                    effective = _corrected_result(original, "categorical", contradiction_code)
            elif original.semantic_type == "numeric_measurement":
                structured = _structured_string_contradiction(series)
                if structured is not None:
                    corrected_type, evidence = structured
                    contradiction_code = f"{evidence['pattern']}_contradicts_numeric_measurement"
                    effective = _corrected_result(original, corrected_type, contradiction_code)

        changed = effective != original
        correction_count += int(changed)
        effective_columns.append(effective)
        audit_entries.append({
            "column_name": original.column_name,
            "contradiction_detected": changed,
            "contradiction_code": contradiction_code,
            "evidence": evidence,
            "original_llm_result": original.model_dump(mode="json"),
            "effective_semantic_result": effective.model_dump(mode="json"),
        })

    telemetry = dict(profile.telemetry)
    telemetry["semantic_contradiction_guard"] = {
        "guard_version": GUARD_VERSION,
        "deterministic": True,
        "evaluated_column_count": len(effective_columns),
        "correction_count": correction_count,
        "ambiguous_shape_overrides_enabled": False,
        "entries": audit_entries,
    }
    return profile.model_copy(update={"columns": effective_columns, "telemetry": telemetry})
