"""Domain-independent deterministic detection of data-quality observations."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from pipelines.pipeline1_cleaning.profiling.column_profiler import make_serializable
from pipelines.pipeline1_cleaning.quality.quality_report import DetectedIssue
from pipelines.pipeline1_cleaning.techniques.privacy_masking import (
    PII_DEFAULT_STRATEGIES,
)


@dataclass(frozen=True)
class DetectionConfig:
    example_cap: int = 5
    categorical_max_unique: int = 50
    categorical_max_unique_ratio: float = 0.20
    categorical_min_non_null: int = 20
    rare_category_max_fraction: float = 0.01
    numeric_parse_min_fraction: float = 0.20
    numeric_parse_max_fraction: float = 0.80
    numeric_parse_min_non_null: int = 5
    date_parse_min_fraction: float = 0.50
    date_parse_min_non_null: int = 5
    outlier_iqr_multiplier: float = 1.5
    outlier_min_unique: int = 10
    outlier_min_unique_ratio: float = 0.05
    pii_pattern_min_fraction: float = 0.80
    pii_semantic_min_confidence: float = 0.0


DEFAULT_CONFIG = DetectionConfig()

EMAIL_PATTERN = re.compile(
    r"^[A-Z0-9.!#$%&'*+/=?^_{}|~-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)+$",
    re.IGNORECASE,
)
NATIONAL_ID_PATTERNS = (
    re.compile(r"^\d{3}-\d{2}-\d{4}$"),
    re.compile(r"^\d{4}[ -]\d{4}[ -]\d{4}$"),
    re.compile(r"^[A-Z]{5}\d{4}[A-Z]$", re.IGNORECASE),
)
PHONE_PATTERNS = (
    re.compile(r"^\+\d{1,3}[ -]\d{3,5}[ -]\d{4,5}$"),
    re.compile(r"^\+\d{1,3}[ -]\d{2,4}[ -]\d{3,4}[ -]\d{4}$"),
    re.compile(r"^\(?\d{3}\)?[ -]\d{3}[ -]\d{4}$"),
)
PERSON_NAME_CONCEPT = re.compile(
    r"(?:\b(?:full|first|last|given|family)[ _-]?name\b|"
    r"\b(?:person|patient|customer|employee|student|client|member|"
    r"applicant|passenger|recipient|sender|contact)(?:'s)?[ _-]+names?\b|"
    r"\bnames?\s+of\s+(?:an?\s+|the\s+)?(?:person|patient|customer|"
    r"employee|student|client|member|applicant|passenger|recipient|sender|"
    r"contact)s?\b)",
    re.IGNORECASE,
)
def _issue_id(issue_type: str, columns: Iterable[str]) -> str:
    identity = issue_type + "|" + "|".join(columns)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"issue-{issue_type}-{digest}"


def _percentage(count: int, denominator: int) -> float:
    return round(count / denominator * 100, 4) if denominator else 0.0


def _examples(series: pd.Series, mask: pd.Series, cap: int) -> List[Any]:
    return [make_serializable(value) for value in series[mask].head(cap).tolist()]


def _is_textual(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
    )


def _date_format(value: str) -> str | None:
    patterns = (
        (r"^\d{4}-\d{2}-\d{2}$", "YYYY-MM-DD"),
        (r"^\d{4}/\d{2}/\d{2}$", "YYYY/MM/DD"),
        (r"^\d{2}/\d{2}/\d{4}$", "NN/NN/YYYY"),
        (r"^\d{2}-\d{2}-\d{4}$", "NN-NN-YYYY"),
        (r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$", "ISO-datetime"),
    )
    for pattern, label in patterns:
        if re.fullmatch(pattern, value.strip()):
            return label
    return None


def _semantic_pii_subtype(
    semantic: Dict[str, Any],
    minimum_confidence: float,
) -> str | None:
    confidence = float(semantic.get("confidence") or 0.0)
    if confidence < minimum_confidence:
        return None
    if (
        semantic.get("semantic_type") == "contact_information"
        or semantic.get("semantic_role") == "contact_field"
        or semantic.get("value_function") == "contact"
    ):
        return "semantic_contact_information"
    concept = str(semantic.get("concept_description") or "")
    if (
        semantic.get("semantic_type") in {"text", "identifier"}
        and PERSON_NAME_CONCEPT.search(concept)
    ):
        return "semantic_person_name"
    return None


def _phone_mask(values: pd.Series) -> pd.Series:
    matched = pd.Series(False, index=values.index)
    for pattern in PHONE_PATTERNS:
        matched |= values.str.fullmatch(pattern, na=False)
    digit_counts = values.str.replace(r"\D", "", regex=True).str.len()
    return matched & digit_counts.between(7, 15)


def detect_issues(
    dataframe: pd.DataFrame,
    dataset_profile: Dict[str, Any],
    config: DetectionConfig = DEFAULT_CONFIG,
    column_semantics: Dict[str, Dict[str, Any]] | None = None,
) -> List[DetectedIssue]:
    issues: List[DetectedIssue] = []
    pii_columns: set[str] = set()
    row_count = len(dataframe)

    def add(
        issue_type: str,
        columns: List[str],
        count: int,
        denominator: int,
        evidence: Dict[str, Any],
        examples: List[Any],
        status: str,
        risk_hint: str = "context_dependent",
    ) -> None:
        issues.append(DetectedIssue(
            issue_id=_issue_id(issue_type, columns),
            issue_type=issue_type,
            target_columns=columns,
            affected_count=count,
            affected_percentage=_percentage(count, denominator),
            evidence=evidence,
            examples=examples[:config.example_cap],
            status=status,
            risk_hint=risk_hint,
        ))

    # Dataset-level exact duplicates use pandas' conventional definition:
    # every matching row after the first occurrence is counted as duplicate.
    duplicate_mask = dataframe.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        duplicate_indices = [make_serializable(index) for index in dataframe.index[duplicate_mask]]
        add(
            "exact_duplicate_rows", [], duplicate_count, row_count,
            {"definition": "Rows equal across all columns after the first occurrence."},
            duplicate_indices, "issue", "context_dependent",
        )

    profile_by_column = {
        item["column_name"]: item for item in dataset_profile["column_profiles"]
    }
    for raw_name in dataframe.columns:
        name = str(raw_name)
        series = dataframe[raw_name]
        profile = profile_by_column[name]
        missing_count = int(profile["missing"]["count"])
        if missing_count:
            add(
                "missing_values", [name], missing_count, row_count,
                {"null_detection": "pandas.isna", "non_null_count": int(series.notna().sum())},
                [make_serializable(index) for index in dataframe.index[series.isna()]],
                "issue",
            )

        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric = pd.to_numeric(series, errors="coerce")
            infinity_mask = pd.Series(np.isinf(numeric.to_numpy(dtype=float)), index=series.index)
            infinity_count = int(infinity_mask.sum())
            if infinity_count:
                add(
                    "non_finite_numeric", [name], infinity_count, row_count,
                    {
                        "positive_infinity_count": int((numeric == np.inf).sum()),
                        "negative_infinity_count": int((numeric == -np.inf).sum()),
                        "nan_handled_as_missing": True,
                    },
                    _examples(series, infinity_mask, config.example_cap), "issue", "medium",
                )

            finite = numeric[np.isfinite(numeric)]
            non_null_count = int(series.notna().sum())
            unique_count = int(finite.nunique())
            unique_ratio = unique_count / non_null_count if non_null_count else 0.0
            if (
                unique_count >= config.outlier_min_unique
                and unique_ratio >= config.outlier_min_unique_ratio
            ):
                q1, q3 = finite.quantile([0.25, 0.75])
                iqr = q3 - q1
                if pd.notna(iqr) and float(iqr) > 0:
                    lower = float(q1 - config.outlier_iqr_multiplier * iqr)
                    upper = float(q3 + config.outlier_iqr_multiplier * iqr)
                    mask = numeric.lt(lower) | numeric.gt(upper)
                    count = int(mask.sum())
                    if count:
                        add(
                            "statistical_outlier_candidate", [name], count, len(finite),
                            {
                                "method": "IQR",
                                "iqr_multiplier": config.outlier_iqr_multiplier,
                                "q1": make_serializable(q1),
                                "q3": make_serializable(q3),
                                "lower_bound": lower,
                                "upper_bound": upper,
                            },
                            _examples(series, mask, config.example_cap), "candidate",
                        )
            continue

        if not _is_textual(series):
            continue

        non_null = series.dropna().astype(str)
        if non_null.empty:
            continue
        stripped = non_null.str.strip()

        national_mask = pd.Series(False, index=stripped.index)
        for pattern in NATIONAL_ID_PATTERNS:
            national_mask |= stripped.str.fullmatch(pattern, na=False)
        pattern_masks = {
            "national_id": national_mask,
            "email": stripped.str.fullmatch(EMAIL_PATTERN, na=False),
            "phone_number": _phone_mask(stripped) & ~national_mask,
        }
        subtype = next(
            (
                candidate
                for candidate in ("national_id", "email", "phone_number")
                if float(pattern_masks[candidate].mean())
                >= config.pii_pattern_min_fraction
            ),
            None,
        )
        detection_basis = "direct_value_pattern"
        matched_count = int(pattern_masks[subtype].sum()) if subtype else 0
        if subtype is None and column_semantics is not None:
            subtype = _semantic_pii_subtype(
                column_semantics.get(name, {}),
                config.pii_semantic_min_confidence,
            )
            detection_basis = "existing_semantic_classification"
            matched_count = len(non_null) if subtype else 0
        if subtype is not None:
            pii_columns.add(name)
            add(
                "pii_candidate",
                [name],
                matched_count,
                len(non_null),
                {
                    "pii_subtype": subtype,
                    "detection_basis": detection_basis,
                    "match_fraction": round(matched_count / len(non_null), 6),
                    "default_masking_strategy": PII_DEFAULT_STRATEGIES[subtype],
                    "privacy_reason": "masked for privacy",
                },
                [],
                "candidate",
                "high",
            )
        leading = non_null.str.match(r"^\s+")
        trailing = non_null.str.match(r".*\s+$")
        whitespace_only = stripped.eq("")
        whitespace_mask = leading | trailing | whitespace_only
        if whitespace_mask.any():
            add(
                "whitespace_issue", [name], int(whitespace_mask.sum()), len(non_null),
                {
                    "leading_count": int(leading.sum()),
                    "trailing_count": int(trailing.sum()),
                    "whitespace_only_count": int(whitespace_only.sum()),
                },
                _examples(non_null, whitespace_mask, config.example_cap), "issue", "low",
            )

        unique_raw = int(non_null.nunique())
        unique_ratio = unique_raw / len(non_null)
        categorical_suitable = (
            len(non_null) >= config.categorical_min_non_null
            and 2 <= unique_raw <= config.categorical_max_unique
            and unique_ratio <= config.categorical_max_unique_ratio
        )
        if categorical_suitable:
            normalized = stripped.str.casefold()
            variants: Dict[str, List[str]] = {}
            for normalized_value, group in non_null.groupby(normalized):
                raw_values = sorted(set(group.tolist()))
                if normalized_value and len(raw_values) > 1:
                    variants[normalized_value] = raw_values[:config.example_cap]
            if variants:
                affected_mask = normalized.isin(variants)
                add(
                    "case_variant_candidate", [name], int(affected_mask.sum()), len(non_null),
                    {"normalization_probe": "strip_then_casefold", "variant_groups": variants},
                    _examples(non_null, affected_mask, config.example_cap), "candidate",
                )

            counts = non_null.value_counts()
            rare_limit = max(1, math.floor(len(non_null) * config.rare_category_max_fraction))
            rare = counts[counts <= rare_limit]
            if not rare.empty:
                rare_mask = non_null.isin(rare.index)
                add(
                    "rare_category_observation", [name], int(rare_mask.sum()), len(non_null),
                    {
                        "maximum_frequency": rare_limit,
                        "maximum_fraction": config.rare_category_max_fraction,
                        "rare_value_count": int(len(rare)),
                    },
                    _examples(non_null, rare_mask, config.example_cap), "observation",
                )

        parsed_numeric = pd.to_numeric(stripped, errors="coerce")
        parseable_mask = parsed_numeric.notna()
        parse_fraction = float(parseable_mask.mean())
        zero_padded = stripped.str.fullmatch(r"[+-]?0\d+").fillna(False)
        if (
            len(non_null) >= config.numeric_parse_min_non_null
            and config.numeric_parse_min_fraction <= parse_fraction <= config.numeric_parse_max_fraction
            and not (parseable_mask.any() and zero_padded[parseable_mask].mean() >= 0.5)
        ):
            add(
                "mixed_numeric_representation_candidate", [name], int(parseable_mask.sum()), len(non_null),
                {
                    "parseable_count": int(parseable_mask.sum()),
                    "unparseable_count": int((~parseable_mask).sum()),
                    "parseable_fraction": round(parse_fraction, 6),
                    "zero_padded_identifier_guard": True,
                },
                _examples(non_null, ~parseable_mask, config.example_cap), "candidate",
            )

        date_formats = stripped.map(_date_format)
        date_like = date_formats.notna()
        date_fraction = float(date_like.mean())
        formats = sorted(set(date_formats.dropna().tolist()))
        if (
            len(non_null) >= config.date_parse_min_non_null
            and date_fraction >= config.date_parse_min_fraction
            and len(formats) > 1
        ):
            add(
                "inconsistent_date_format_candidate", [name], int(date_like.sum()), len(non_null),
                {
                    "recognized_formats": formats,
                    "date_like_fraction": round(date_fraction, 6),
                    "method": "strict_shape_recognition",
                },
                _examples(non_null, date_like, config.example_cap), "candidate",
            )

    # Once a column is privacy-sensitive, do not retain its raw values in
    # examples belonging to any overlapping quality observation.
    for issue in issues:
        if pii_columns.intersection(issue.target_columns):
            issue.examples = []
    return issues
