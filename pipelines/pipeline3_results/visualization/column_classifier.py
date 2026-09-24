"""Deterministic Column Intelligence for Pipeline 3 — Results.

This module provides deterministic, auditable, and model-independent column
classification for tabular datasets. Each column is mapped to exactly one of
the seven analytical roles:
- MEASURE
- CATEGORY
- TEMPORAL
- IDENTIFIER
- DESCRIPTIVE_TEXT
- BOOLEAN
- UNKNOWN

Every classification is explained through explicit signal tokens without
invented confidence scores, non-deterministic heuristics, or external dependencies.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pipelines.pipeline3_results.visualization.schemas import (
    ALL_ROLES,
    ColumnClassification,
    ColumnRole,
    DatasetClassification,
)

# ---------------------------------------------------------------------------
# Regex patterns & token vocabularies
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})"
    r"(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)

_EXACT_IDENTIFIER_NAMES = {
    "id", "uuid", "guid", "pk", "sk", "key", "row_id", "row_num",
    "index", "record_id", "txn_id", "tx_id",
}
_IDENTIFIER_SUFFIXES = ("id", "uuid", "guid", "key", "identifier", "token", "hash")
_IDENTIFIER_PREFIXES = ("id_", "uuid_", "guid_", "key_")

_BOOLEAN_PREFIXES = ("is_", "has_", "was_", "can_", "should_", "did_", "does_")
_BOOLEAN_SUFFIXES = ("_flag", "_flg", "_ind", "_indicator", "_bool", "_boolean", "_binary")
_BOOLEAN_EXACT_TERMS = {
    "active", "churned", "defaulted", "subscribed", "survived",
    "opt_in", "opted_in", "enabled", "approved", "verified",
    "selected", "completed", "cancelled", "canceled", "paid",
    "flag", "indicator", "binary", "target_binary",
}

_TEMPORAL_TERMS = {
    "date", "time", "timestamp", "datetime", "year", "month", "day",
    "quarter", "hour", "minute", "second", "epoch", "dob", "created",
    "updated", "modified", "registered", "period", "dt", "ts",
}
_DURATION_TERMS = {
    "duration", "spent", "elapsed", "length", "seconds", "minutes",
    "hours", "days", "months", "weeks",
}

_MEASURE_TERMS = {
    "amount", "count", "sum", "qty", "quantity", "total", "num",
    "number", "rate", "score", "revenue", "sales", "price", "cost",
    "fee", "salary", "wage", "duration", "spent", "distance", "weight",
    "height", "age", "pct", "percent", "percentage", "ratio",
    "balance", "profit", "loss", "budget", "margin", "tax", "discount",
    "value", "income", "gdp", "val", "latitude", "longitude", "lat", "lon",
    "temp", "temperature", "size", "speed", "volume", "depth", "width",
    "length", "fare", "payment", "spend", "expenses",
}

_DESCRIPTIVE_TERMS = {
    "desc", "description", "comment", "comments", "review", "reviews",
    "note", "notes", "text", "summary", "abstract", "feedback",
    "content", "body", "message", "details", "narrative", "remarks",
    "transcript", "biography", "bio", "explanation", "reason", "detail",
}

_CATEGORY_TERMS = {
    "category", "type", "class", "group", "status", "segment",
    "department", "genre", "region", "country", "city", "state",
    "province", "district", "zone", "tier", "grade", "level",
    "priority", "gender", "sex", "race", "ethnicity", "marital",
    "brand", "model", "vendor", "channel", "source", "method",
    "role", "color", "colour", "division", "sector", "industry",
    "style", "kind", "mode", "stage",
}


# ---------------------------------------------------------------------------
# Helpers for name token analysis & evidence extraction
# ---------------------------------------------------------------------------

def _normalize_name_tokens(col_name: str) -> list[str]:
    """Tokenize column name by splitting camelCase, underscores, and non-alphanumerics."""
    s = str(col_name).strip()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.casefold()).strip("_")
    return [t for t in s.split("_") if t]


def _is_identifier_name(tokens: list[str], col_name: str) -> bool:
    norm = "_".join(tokens)
    if norm in _EXACT_IDENTIFIER_NAMES:
        return True
    if tokens:
        if tokens[-1] in _IDENTIFIER_SUFFIXES:
            return True
        if tokens[0] in _IDENTIFIER_PREFIXES or tokens[0] in {"id", "uuid", "guid"}:
            return True
    if re.search(
        r"(?:customer|account|patient|employee|student|user|order|transaction|record|row|item|product|client|txn|session)id$",
        norm,
    ):
        return True
    return False


def _is_boolean_name(tokens: list[str], col_name: str) -> bool:
    norm = "_".join(tokens)
    if norm in _BOOLEAN_EXACT_TERMS:
        return True
    for p in _BOOLEAN_PREFIXES:
        if norm.startswith(p):
            return True
    for s in _BOOLEAN_SUFFIXES:
        if norm.endswith(s):
            return True
    if tokens and tokens[0] in {"is", "has", "was", "can", "should", "did", "does"}:
        return True
    if tokens and tokens[-1] in {"flag", "flg", "ind", "indicator", "bool", "boolean", "binary"}:
        return True
    return False


def _is_temporal_name(tokens: list[str], col_name: str) -> bool:
    if any(t in _DURATION_TERMS for t in tokens) and not any(
        t in {"date", "dob", "created", "timestamp", "datetime"} for t in tokens
    ):
        return False
    return any(t in _TEMPORAL_TERMS for t in tokens)


def _is_measure_name(tokens: list[str], col_name: str) -> bool:
    return any(t in _MEASURE_TERMS for t in tokens)


def _is_descriptive_name(tokens: list[str], col_name: str) -> bool:
    return any(t in _DESCRIPTIVE_TERMS for t in tokens)


def _is_category_name(tokens: list[str], col_name: str) -> bool:
    return any(t in _CATEGORY_TERMS for t in tokens)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        return dumped if isinstance(dumped, Mapping) else {}
    return {}


def _extract_column_evidence(
    pipeline1_report: Any, col_name: str
) -> dict[str, Any]:
    """Safely extract semantic and profiling evidence for a column from Pipeline 1 output."""
    if not pipeline1_report:
        return {}
    report = _mapping(pipeline1_report)
    col_cf = str(col_name).casefold()

    extracted: dict[str, Any] = {}

    # Check semantic profile
    semantic_profile = _mapping(report.get("semantic_profile", {}))
    if not semantic_profile and "columns" in report:
        semantic_profile = report

    columns_entry = semantic_profile.get("columns", [])
    if isinstance(columns_entry, list):
        for item in columns_entry:
            c_dict = _mapping(item)
            c_name = str(c_dict.get("column_name", "")).casefold()
            if c_name == col_cf:
                extracted.update(c_dict)
                break
    elif isinstance(columns_entry, Mapping):
        for c_key, c_val in columns_entry.items():
            if str(c_key).casefold() == col_cf:
                extracted.update(_mapping(c_val))
                break

    # Also check before_profile / after_profile column profiles
    for prof_key in ("after_profile", "before_profile"):
        prof = _mapping(report.get(prof_key, {}))
        col_profs = _mapping(prof.get("column_profiles", {}))
        for c_key, c_val in col_profs.items():
            if str(c_key).casefold() == col_cf:
                c_data = _mapping(c_val)
                for k in ("physical_dtype", "sample_values", "numeric_statistics"):
                    if k in c_data and k not in extracted:
                        extracted[k] = c_data[k]
                break

    return extracted


def _date_parseable_fraction(sample: pd.Series) -> float:
    """Test date parseability on a series, returning fraction in [0.0, 1.0]."""
    if len(sample) == 0:
        return 0.0
    str_sample = sample.astype(str).str.strip()
    regex_matches = float(str_sample.str.match(_DATE_RE).mean())
    if regex_matches >= 0.75:
        return regex_matches

    # Try pd.to_datetime on a sample of up to 100 values
    test_sub = str_sample.iloc[:100]
    try:
        parsed = pd.to_datetime(test_sub, errors="coerce", format="mixed")
        return float(parsed.notna().mean())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Core Classification Engine
# ---------------------------------------------------------------------------

def classify_column(
    series: pd.Series,
    column_evidence: Mapping[str, Any] | None = None,
    semantic_evidence: Mapping[str, Any] | None = None,
) -> ColumnClassification:
    """Deterministically classify an individual column into one of 7 analytical roles.

    Parameters
    ----------
    series:
        The pandas Series to inspect.
    column_evidence:
        Optional structural or profiling evidence from Pipeline 1.
    semantic_evidence:
        Optional semantic profile dictionary from Pipeline 1.

    Returns
    -------
    ColumnClassification
        The deterministic classification with explicit explanation signals.
    """
    col_name = str(series.name) if series.name is not None else "column"
    dtype_str = str(series.dtype)
    total_rows = len(series)

    # Consolidate evidence safely
    evidence: dict[str, Any] = {}
    if column_evidence:
        evidence.update(_mapping(column_evidence))
    if semantic_evidence:
        evidence.update(_mapping(semantic_evidence))

    # Name tokens and name signals
    tokens = _normalize_name_tokens(col_name)
    is_id_name = _is_identifier_name(tokens, col_name)
    is_bool_name = _is_boolean_name(tokens, col_name)
    is_temporal_name = _is_temporal_name(tokens, col_name)
    is_measure_name = _is_measure_name(tokens, col_name)
    is_desc_name = _is_descriptive_name(tokens, col_name)
    is_cat_name = _is_category_name(tokens, col_name)

    # Pipeline 1 evidence signals
    sem_type = str(evidence.get("semantic_type", "")).casefold()
    val_func = str(evidence.get("value_function", "")).casefold()
    num_meaning = str(evidence.get("numeric_meaning", "")).casefold()
    sem_role = str(evidence.get("semantic_role", "")).casefold()

    p1_is_bool = (
        val_func == "boolean_state"
        or sem_type in {"boolean", "flag", "indicator"}
        or sem_role == "binary_target"
    )
    p1_is_temporal = (
        sem_type in {"date", "time", "datetime", "timestamp", "temporal"}
        or val_func in {"temporal", "time_index", "date", "timestamp"}
    )
    p1_is_id = (
        sem_type in {"identifier", "contact_information"}
        or sem_role in {"primary_identifier", "foreign_identifier", "transaction_identifier"}
        or val_func == "identifier"
    )
    p1_is_measure = (
        sem_type in {"currency", "measurement", "metric", "continuous_quantity", "count"}
        or num_meaning in {
            "continuous_quantity", "count", "currency", "percentage",
            "ratio", "physical_measurement",
        }
        or val_func in {"continuous_quantity", "discrete_count", "ratio"}
        or sem_role == "continuous_target"
    )
    p1_is_text = (
        sem_type in {"text", "free_text", "narrative", "description"}
        or val_func in {"narrative_text", "free_text"}
    )
    p1_is_cat = (
        sem_type in {"category", "categorical", "nominal", "ordinal"}
        or val_func in {"category_nominal", "category_ordinal"}
    )

    # -----------------------------------------------------------------------
    # Rule 0: Empty Series
    # -----------------------------------------------------------------------
    if total_rows == 0:
        return ColumnClassification(
            name=col_name,
            role="UNKNOWN",
            dtype=dtype_str,
            unique_count=0,
            unique_ratio=0.0,
            missing_count=0,
            missing_ratio=0.0,
            signals=["empty_column"],
        )

    # -----------------------------------------------------------------------
    # Rule 1: All values missing
    # -----------------------------------------------------------------------
    non_null = series.dropna()
    non_null_count = len(non_null)
    missing_count = total_rows - non_null_count
    missing_ratio = round(missing_count / total_rows, 4)

    if non_null_count == 0:
        return ColumnClassification(
            name=col_name,
            role="UNKNOWN",
            dtype=dtype_str,
            unique_count=0,
            unique_ratio=0.0,
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            signals=["all_values_missing"],
        )

    unique_count = int(non_null.nunique())
    unique_ratio = round(unique_count / non_null_count, 4)

    # -----------------------------------------------------------------------
    # Rule 2: Single constant value (zero variance -> UNKNOWN for visualization)
    # -----------------------------------------------------------------------
    if unique_count == 1:
        return ColumnClassification(
            name=col_name,
            role="UNKNOWN",
            dtype=dtype_str,
            unique_count=1,
            unique_ratio=unique_ratio,
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            signals=["constant_single_value", "insufficient_variance"],
        )

    # -----------------------------------------------------------------------
    # Rule 3: Physical Boolean dtype
    # -----------------------------------------------------------------------
    if pd.api.types.is_bool_dtype(non_null.dtype):
        signals = ["boolean_dtype"]
        if is_bool_name:
            signals.append("boolean_name_signal")
        if p1_is_bool:
            signals.append("pipeline1_boolean_semantic")
        return ColumnClassification(
            name=col_name,
            role="BOOLEAN",
            dtype=dtype_str,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            signals=signals,
        )

    # -----------------------------------------------------------------------
    # Rule 4: Physical Datetime / Period dtype
    # -----------------------------------------------------------------------
    if pd.api.types.is_datetime64_any_dtype(non_null.dtype) or isinstance(
        non_null.dtype, pd.PeriodDtype
    ):
        signals = ["datetime_dtype"]
        if is_temporal_name:
            signals.append("temporal_name_signal")
        if p1_is_temporal:
            signals.append("pipeline1_temporal_semantic")
        return ColumnClassification(
            name=col_name,
            role="TEMPORAL",
            dtype=dtype_str,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            signals=signals,
        )

    # -----------------------------------------------------------------------
    # Rule 5: String / Categorical Boolean values
    # -----------------------------------------------------------------------
    if unique_count == 2 and not pd.api.types.is_numeric_dtype(non_null.dtype):
        str_vals = set(non_null.astype(str).str.strip().str.lower().unique())
        if str_vals in ({"true", "false"}, {"t", "f"}, {"yes", "no"}, {"y", "n"}):
            signals = ["string_boolean_values"]
            if is_bool_name:
                signals.append("boolean_name_signal")
            if p1_is_bool:
                signals.append("pipeline1_boolean_semantic")
            return ColumnClassification(
                name=col_name,
                role="BOOLEAN",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

    # -----------------------------------------------------------------------
    # Rule 6: Numeric Binary {0, 1}
    # Note: As instructed, do NOT classify every numeric {0, 1} as BOOLEAN.
    # Require boolean semantics from column name or Pipeline 1 evidence.
    # Otherwise, classify conservatively as CATEGORY.
    # -----------------------------------------------------------------------
    if pd.api.types.is_numeric_dtype(non_null.dtype) and unique_count == 2:
        num_vals = set(non_null.unique())
        if num_vals.issubset({0, 1, 0.0, 1.0}):
            if is_bool_name or p1_is_bool:
                signals = ["numeric_binary_values"]
                if is_bool_name:
                    signals.append("boolean_name_signal")
                if p1_is_bool:
                    signals.append("pipeline1_boolean_semantic")
                return ColumnClassification(
                    name=col_name,
                    role="BOOLEAN",
                    dtype=dtype_str,
                    unique_count=unique_count,
                    unique_ratio=unique_ratio,
                    missing_count=missing_count,
                    missing_ratio=missing_ratio,
                    signals=signals,
                )
            else:
                signals = [
                    "numeric_binary_values",
                    "no_boolean_semantics_fallback_category",
                ]
                if is_cat_name:
                    signals.append("category_name_signal")
                return ColumnClassification(
                    name=col_name,
                    role="CATEGORY",
                    dtype=dtype_str,
                    unique_count=unique_count,
                    unique_ratio=unique_ratio,
                    missing_count=missing_count,
                    missing_ratio=missing_ratio,
                    signals=signals,
                )

    # -----------------------------------------------------------------------
    # Rule 7: UUID pattern match
    # -----------------------------------------------------------------------
    if not pd.api.types.is_numeric_dtype(non_null.dtype):
        sample_str = non_null.astype(str).str.strip().iloc[:500]
        uuid_matches = float(sample_str.str.fullmatch(_UUID_RE).mean())
        if uuid_matches >= 0.85:
            signals = ["uuid_pattern_match"]
            if is_id_name:
                signals.append("identifier_name_signal")
            if p1_is_id:
                signals.append("pipeline1_identifier_semantic")
            return ColumnClassification(
                name=col_name,
                role="IDENTIFIER",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

    # -----------------------------------------------------------------------
    # Rule 8: Pipeline 1 Strong Semantic Identifier
    # -----------------------------------------------------------------------
    if p1_is_id:
        signals = ["pipeline1_identifier_semantic"]
        if is_id_name:
            signals.append("identifier_name_signal")
        if unique_ratio >= 0.5:
            signals.append("high_cardinality")
        return ColumnClassification(
            name=col_name,
            role="IDENTIFIER",
            dtype=dtype_str,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            signals=signals,
        )

    # -----------------------------------------------------------------------
    # Rule 9: Identifier by Name + High Uniqueness
    # -----------------------------------------------------------------------
    if is_id_name:
        # If column name signals an ID and uniqueness is sufficient
        if unique_ratio >= 0.30 or non_null_count <= 20 or not is_measure_name:
            signals = ["identifier_name_signal"]
            if unique_ratio >= 0.80:
                signals.append("high_unique_ratio")
            return ColumnClassification(
                name=col_name,
                role="IDENTIFIER",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

    # -----------------------------------------------------------------------
    # Rule 10: Temporal (Conservative: requires temporal name OR P1 semantic)
    # -----------------------------------------------------------------------
    if is_temporal_name or p1_is_temporal:
        if pd.api.types.is_numeric_dtype(non_null.dtype):
            min_val = float(non_null.min())
            max_val = float(non_null.max())
            is_int_like = bool((non_null % 1 == 0).all())
            # Integer year column
            if (
                is_int_like
                and 1800 <= min_val <= 2100
                and 1800 <= max_val <= 2100
                and ("year" in tokens or "yr" in tokens)
            ):
                signals = ["year_integer_range"]
                if is_temporal_name:
                    signals.append("temporal_name_signal")
                if p1_is_temporal:
                    signals.append("pipeline1_temporal_semantic")
                return ColumnClassification(
                    name=col_name,
                    role="TEMPORAL",
                    dtype=dtype_str,
                    unique_count=unique_count,
                    unique_ratio=unique_ratio,
                    missing_count=missing_count,
                    missing_ratio=missing_ratio,
                    signals=signals,
                )
            # Epoch timestamp column
            if (
                min_val >= 946684800
                and max_val <= 2500000000
                and any(t in {"timestamp", "epoch", "time", "ts"} for t in tokens)
            ):
                signals = ["epoch_timestamp"]
                if is_temporal_name:
                    signals.append("temporal_name_signal")
                return ColumnClassification(
                    name=col_name,
                    role="TEMPORAL",
                    dtype=dtype_str,
                    unique_count=unique_count,
                    unique_ratio=unique_ratio,
                    missing_count=missing_count,
                    missing_ratio=missing_ratio,
                    signals=signals,
                )
        else:
            parse_fraction = _date_parseable_fraction(non_null)
            if parse_fraction >= 0.75:
                signals = ["date_parseable_format"]
                if is_temporal_name:
                    signals.append("temporal_name_signal")
                if p1_is_temporal:
                    signals.append("pipeline1_temporal_semantic")
                return ColumnClassification(
                    name=col_name,
                    role="TEMPORAL",
                    dtype=dtype_str,
                    unique_count=unique_count,
                    unique_ratio=unique_ratio,
                    missing_count=missing_count,
                    missing_ratio=missing_ratio,
                    signals=signals,
                )

    # -----------------------------------------------------------------------
    # Rule 11: Descriptive Text
    # -----------------------------------------------------------------------
    if not pd.api.types.is_numeric_dtype(non_null.dtype):
        str_series = non_null.astype(str).str.strip()
        words = str_series.str.split()
        mean_wc = float(words.str.len().mean())
        mean_len = float(str_series.str.len().mean())
        max_len = int(str_series.str.len().max())

        if p1_is_text:
            signals = ["pipeline1_text_semantic"]
            return ColumnClassification(
                name=col_name,
                role="DESCRIPTIVE_TEXT",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

        if is_desc_name and (mean_wc >= 2.0 or mean_len >= 15 or max_len >= 40):
            signals = ["descriptive_name_signal", "free_text_content"]
            return ColumnClassification(
                name=col_name,
                role="DESCRIPTIVE_TEXT",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

        if mean_wc >= 4.0 and mean_len >= 30 and max_len >= 50 and unique_ratio >= 0.50:
            signals = ["high_word_count", "high_text_length"]
            return ColumnClassification(
                name=col_name,
                role="DESCRIPTIVE_TEXT",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

    # -----------------------------------------------------------------------
    # Rule 12: High-cardinality Alphanumeric Identifiers
    # -----------------------------------------------------------------------
    if (
        not pd.api.types.is_numeric_dtype(non_null.dtype)
        and unique_ratio >= 0.85
        and non_null_count >= 10
    ):
        str_series = non_null.astype(str).str.strip()
        mean_len = float(str_series.str.len().mean())
        has_no_spaces = float((~str_series.str.contains(r"\s")).mean()) >= 0.90
        if mean_len <= 35 and has_no_spaces:
            signals = ["high_uniqueness_code", "short_alphanumeric_identifier"]
            if is_id_name:
                signals.append("identifier_name_signal")
            return ColumnClassification(
                name=col_name,
                role="IDENTIFIER",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

    # -----------------------------------------------------------------------
    # Rule 13: Numeric Data Types (MEASURE vs CATEGORY)
    # -----------------------------------------------------------------------
    if pd.api.types.is_numeric_dtype(non_null.dtype):
        # Pipeline 1 measure semantic
        if p1_is_measure:
            signals = ["numeric_dtype", "pipeline1_measure_semantic"]
            return ColumnClassification(
                name=col_name,
                role="MEASURE",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

        # Explicit measure name signal
        if is_measure_name:
            signals = ["numeric_dtype", "measure_name_signal"]
            return ColumnClassification(
                name=col_name,
                role="MEASURE",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

        # Float with real continuous fractional components
        has_fraction = bool((non_null % 1 != 0).any())
        if has_fraction:
            signals = ["numeric_float_continuous_values"]
            return ColumnClassification(
                name=col_name,
                role="MEASURE",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

        # High-cardinality numeric integers
        if unique_count > 15:
            signals = ["numeric_high_cardinality_measure"]
            return ColumnClassification(
                name=col_name,
                role="MEASURE",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

        # Low-cardinality discrete integers (e.g. 2 <= unique_count <= 15)
        if is_cat_name:
            signals = ["discrete_numeric_low_cardinality", "category_name_signal"]
            return ColumnClassification(
                name=col_name,
                role="CATEGORY",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

        if unique_ratio <= 0.25:
            signals = ["discrete_numeric_low_cardinality_category"]
            return ColumnClassification(
                name=col_name,
                role="CATEGORY",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

        # Small sample with varied integers without category hints
        signals = ["numeric_values_measure"]
        return ColumnClassification(
            name=col_name,
            role="MEASURE",
            dtype=dtype_str,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            signals=signals,
        )

    # -----------------------------------------------------------------------
    # Rule 14: Categorical / String Columns (CATEGORY)
    # -----------------------------------------------------------------------
    if not pd.api.types.is_numeric_dtype(non_null.dtype):
        if p1_is_cat:
            signals = ["pipeline1_category_semantic"]
            return ColumnClassification(
                name=col_name,
                role="CATEGORY",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

        if unique_count <= 100 or unique_ratio < 0.80:
            signals = ["bounded_cardinality_category"]
            if is_cat_name:
                signals.append("category_name_signal")
            return ColumnClassification(
                name=col_name,
                role="CATEGORY",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

        # Short nominal labels even if unique_ratio >= 0.8 on small dataset
        str_series = non_null.astype(str).str.strip()
        mean_wc = float(str_series.str.split().str.len().mean())
        max_len = int(str_series.str.len().max())
        if unique_count <= 25 and mean_wc <= 2.5 and max_len <= 35:
            signals = ["short_nominal_tokens_category"]
            if is_cat_name:
                signals.append("category_name_signal")
            return ColumnClassification(
                name=col_name,
                role="CATEGORY",
                dtype=dtype_str,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                signals=signals,
            )

    # -----------------------------------------------------------------------
    # Rule 15: Fallback -> UNKNOWN
    # -----------------------------------------------------------------------
    return ColumnClassification(
        name=col_name,
        role="UNKNOWN",
        dtype=dtype_str,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        missing_count=missing_count,
        missing_ratio=missing_ratio,
        signals=["insufficient_or_conflicting_evidence"],
    )


# ---------------------------------------------------------------------------
# Dataset-level Classification Function
# ---------------------------------------------------------------------------

def classify_columns(
    dataframe: pd.DataFrame,
    pipeline1_report: Mapping[str, Any] | None = None,
    dataset_name: str | None = None,
) -> DatasetClassification:
    """Deterministically classify all columns of a tabular dataset.

    Parameters
    ----------
    dataframe:
        The pandas DataFrame to classify.
    pipeline1_report:
        Optional Pipeline 1 cleaning / profiling report mapping.
    dataset_name:
        Optional dataset name identifier.

    Returns
    -------
    DatasetClassification
        Complete dataset classification with column roles and role counts.
    """
    row_count = len(dataframe)
    column_count = len(dataframe.columns)

    if column_count == 0:
        return DatasetClassification(
            dataset_name=dataset_name,
            row_count=row_count,
            column_count=0,
            columns=[],
            role_counts={r: 0 for r in ALL_ROLES},
        )

    classified_columns: list[ColumnClassification] = []
    for col_name in dataframe.columns:
        series = dataframe[col_name]
        col_evidence = _extract_column_evidence(pipeline1_report, str(col_name))
        col_cls = classify_column(
            series=series,
            column_evidence=col_evidence,
            semantic_evidence=col_evidence,
        )
        classified_columns.append(col_cls)

    role_counts = {
        role: sum(1 for c in classified_columns if c.role == role)
        for role in ALL_ROLES
    }

    return DatasetClassification(
        dataset_name=dataset_name,
        row_count=row_count,
        column_count=column_count,
        columns=classified_columns,
        role_counts=role_counts,
    )


def classify_dataframe(
    dataframe: pd.DataFrame,
    pipeline1_report: Mapping[str, Any] | None = None,
    dataset_name: str | None = None,
) -> DatasetClassification:
    """Convenience alias for :func:`classify_columns`."""
    return classify_columns(
        dataframe=dataframe,
        pipeline1_report=pipeline1_report,
        dataset_name=dataset_name,
    )
