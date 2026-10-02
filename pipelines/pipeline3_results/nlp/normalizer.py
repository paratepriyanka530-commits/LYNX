"""Deterministic, dataset-independent language normalization for Pipeline 3."""

from __future__ import annotations

import re
from typing import TypedDict

from pipelines.pipeline3_results.nlp.schemas import (
    Aggregation,
    ComparisonOperation,
    SharedChartType,
    SharedChartVariant,
)


# Longest phrases come first so "how many" wins before any component token.
AGGREGATION_SYNONYMS: tuple[tuple[str, Aggregation], ...] = (
    ("number of", "COUNT"),
    ("how many", "COUNT"),
    ("average", "MEAN"),
    ("combined", "SUM"),
    ("minimum", "MIN"),
    ("smallest", "MIN"),
    ("maximum", "MAX"),
    ("largest", "MAX"),
    ("median", "MEDIAN"),
    ("count", "COUNT"),
    ("total", "SUM"),
    ("mean", "MEAN"),
    ("sum", "SUM"),
    ("avg", "MEAN"),
    ("min", "MIN"),
    ("max", "MAX"),
)

COMPARISON_SYNONYMS: tuple[tuple[str, ComparisonOperation], ...] = (
    ("highest", "HIGHEST"),
    ("greatest", "HIGHEST"),
    ("lowest", "LOWEST"),
    ("smallest", "LOWEST"),
    ("bottom", "LOWEST"),
    ("worst", "LOWEST"),
    ("least", "LOWEST"),
    ("best", "HIGHEST"),
    ("most", "HIGHEST"),
    ("top", "HIGHEST"),
)

CHART_SYNONYMS: tuple[tuple[str, SharedChartType], ...] = (
    ("doughnut chart", "PIE"),
    ("doughnut graph", "PIE"),
    ("donut chart", "PIE"),
    ("donut graph", "PIE"),
    ("pie chart", "PIE"),
    ("pie graph", "PIE"),
    ("doughnut", "PIE"),
    ("donut", "PIE"),
    ("pie", "PIE"),
    ("frequency chart", "COUNT_BAR"),
    ("count chart", "COUNT_BAR"),
    ("scatter plot", "SCATTER"),
    ("scatter chart", "SCATTER"),
    ("column chart", "BAR"),
    ("bar graph", "BAR"),
    ("bar chart", "BAR"),
    ("line graph", "LINE"),
    ("line chart", "LINE"),
    ("time series", "LINE"),
    ("over time", "LINE"),
    ("histogram", "HISTOGRAM"),
    ("trend", "LINE"),
)


class NormalizedLanguage(TypedDict):
    """Generic concepts only; deliberately contains no column-role decisions."""

    normalized_text: str
    tokens: list[str]
    aggregation: Aggregation
    comparison: ComparisonOperation
    requested_chart: SharedChartType | None
    requested_chart_variant: SharedChartVariant | None


def normalize_text(text: str) -> str:
    """Case-fold text, normalize separators/punctuation, and collapse whitespace."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    cleaned = re.sub(r"[_\-]+", " ", text.casefold())
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return " ".join(cleaned.split())


def tokenize(text: str) -> list[str]:
    """Return deterministic tokens from normalized text."""
    normalized = normalize_text(text)
    return re.findall(r"\b\w+\b", normalized)


def _first_concept(text: str, vocabulary: tuple[tuple[str, object], ...], default: object) -> object:
    normalized = normalize_text(text)
    matches: list[tuple[int, int, object]] = []
    for order, (phrase, concept) in enumerate(vocabulary):
        match = re.search(r"\b" + re.escape(phrase) + r"\b", normalized)
        if match:
            matches.append((match.start(), order, concept))
    return min(matches, key=lambda item: (item[0], item[1]))[2] if matches else default


def normalize_aggregation(text: str) -> Aggregation:
    """Return the first explicit aggregation concept, or ``NONE``."""
    return _first_concept(text, AGGREGATION_SYNONYMS, "NONE")  # type: ignore[return-value]


def normalize_comparison(text: str) -> ComparisonOperation:
    """Return a comparison-language concept without deciding query semantics."""
    return _first_concept(text, COMPARISON_SYNONYMS, "NONE")  # type: ignore[return-value]


def normalize_chart(text: str) -> SharedChartType | None:
    """Return the first explicit generic chart concept, if present."""
    return _first_concept(text, CHART_SYNONYMS, None)  # type: ignore[return-value]


def normalize_chart_variant(text: str) -> SharedChartVariant | None:
    """Preserve the requested presentation variant for a supported chart family."""
    normalized = normalize_text(text)
    if re.search(r"\b(?:doughnut|donut)\b", normalized):
        return "DOUGHNUT"
    return None


def normalize_language(text: str) -> NormalizedLanguage:
    """Expose normalized generic concepts without resolving dataframe columns."""
    normalized = normalize_text(text)
    return {
        "normalized_text": normalized,
        "tokens": tokenize(normalized),
        "aggregation": normalize_aggregation(normalized),
        "comparison": normalize_comparison(normalized),
        "requested_chart": normalize_chart(normalized),
        "requested_chart_variant": normalize_chart_variant(normalized),
    }


__all__ = [
    "AGGREGATION_SYNONYMS",
    "CHART_SYNONYMS",
    "COMPARISON_SYNONYMS",
    "NormalizedLanguage",
    "normalize_aggregation",
    "normalize_chart",
    "normalize_chart_variant",
    "normalize_comparison",
    "normalize_language",
    "normalize_text",
    "tokenize",
]
