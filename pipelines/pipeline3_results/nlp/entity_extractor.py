"""Deterministic, language-only analytical entity extraction for Pipeline 3."""

from __future__ import annotations

import re

from pipelines.pipeline3_results.nlp.normalizer import (
    normalize_aggregation,
    normalize_chart,
    normalize_chart_variant,
    normalize_comparison,
    normalize_text,
)
from pipelines.pipeline3_results.nlp.schemas import (
    ExtractedEntities,
    IntentClassification,
    QueryFilter,
    QueryIntent,
)


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}
_NUMBER_PATTERN = r"(?:\d+|" + "|".join(_NUMBER_WORDS) + r")"

_EXPLICIT_CHART_RE = re.compile(
    r"\b(?:bar\s+(?:graph|chart)|column\s+chart|line\s+(?:graph|chart)|"
    r"scatter\s+(?:plot|chart)|histogram|frequency\s+chart|count\s+chart|"
    r"pie(?:\s+(?:chart|graph))?|(?:doughnut|donut)(?:\s+(?:chart|graph))?)\b"
)
_ANY_CHART_RE = re.compile(
    r"\b(?P<chart>radar|area|bubble|heatmap|box|violin|treemap|sunburst|"
    r"funnel|gauge)\s+(?P<kind>chart|graph|plot)\b"
)
_CHART_SUFFIX_RE = re.compile(
    r"\s+(?:(?:in|as)(?:\s+a)?\s+)?(?:pie|doughnut|donut)"
    r"(?:\s+(?:chart|graph))?\s*$"
)
_TOP_BOTTOM_RE = re.compile(
    rf"\b(?P<direction>top|bottom)\s+(?P<limit>{_NUMBER_PATTERN})\s+"
    r"(?P<dimension>.+?)\s+(?:by|based\s+on)\s+(?P<measure>.+)$"
)
_COMPARATIVE_QUESTION_RE = re.compile(
    r"^(?:which|what)\s+(?P<dimension>.+?)\s+"
    r"(?:has|have|had|generated|produced|recorded|shows?|showed|performed|brought\s+in)\s+"
    r"(?:the\s+)?(?P<comparison>highest|lowest|most|least|best|worst|greatest|smallest)"
    r"(?:\s+(?P<measure>.+))?$"
)
_NATURAL_COMPARATIVE_RE = re.compile(
    r"^(?:which|what(?:\s+is)?)(?:\s+the)?\s+"
    r"(?:(?P<leading>most|least)\s+(?P<verb1>sell|selling|sold|ordered|common)\s+"
    r"(?P<dimension1>.+)|(?P<dimension2>.+?)\s+"
    r"(?P<verb2>sells?|selling|sold|(?:is\s+)?ordered|(?:is\s+)?common)\s+"
    r"(?:the\s+)?(?P<trailing>most|least))$"
)
_BETWEEN_FIELDS_RE = re.compile(
    r"\b(?:relationship|correlation|association)(?:\s+of)?\s+between\s+"
    r"(?P<first>.+?)\s+and\s+(?P<second>.+)$"
)
_EXPLICIT_TWO_FIELD_RE = re.compile(
    r"\b(?:scatter\s+(?:plot|chart)|relationship|correlation)(?:\s+of)?\s+"
    r"(?P<first>.+?)\s+and\s+(?P<second>.+)$"
)
_VERSUS_RE = re.compile(
    r"(?P<first>.+?)\s+(?:versus|vs|against)\s+(?P<second>.+)$"
)
_RELATION_RE = re.compile(
    r"(?P<measure>.+?)\s+(?P<relation>by|across|per|over|for\s+each)\s+(?P<dimension>.+)$"
)
_WHERE_FILTER_RE = re.compile(
    r"(?P<prefix>.+?)\s+where\s+(?P<field>[\w\s]+?)\s+is\s+(?P<value>[\w\s]+)$"
)
_IN_NAMED_FILTER_RE = re.compile(
    r"(?P<prefix>.+?)\s+in\s+(?P<value>\w+(?:\s+\w+)*)\s+(?P<field>\w+)$"
)
_YEAR_FILTER_RE = re.compile(r"(?P<prefix>.+?)\s+(?P<prep>in|for)\s+(?P<year>\d{4})$")

_TIME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bover\s+(?:time|day|week|month|quarter|year)\b"),
    re.compile(r"\bby\s+(?:day|week|month|quarter|year)\b"),
    re.compile(r"\b(?:daily|weekly|monthly|quarterly|yearly|annually)\b"),
    re.compile(r"\b(?:in|for)\s+\d{4}\b"),
    re.compile(r"\b(?:last|this|next)\s+(?:day|week|month|quarter|year)\b"),
    re.compile(r"\bbetween\s+\d{4}\s+and\s+\d{4}\b"),
)

_LEADING_NOISE_RE = re.compile(
    r"^(?:(?:please|show|display|give|make|plot|graph|visualize|visualise|find|tell|"
    r"compare)(?:\s+me)?\s+)+"
)
_CHART_NOISE_RE = re.compile(
    r"\b(?:bar\s+(?:graph|chart)|column\s+chart|line\s+(?:graph|chart)|"
    r"scatter\s+(?:plot|chart)|histogram|frequency\s+chart|count\s+chart|"
    r"pie(?:\s+(?:chart|graph))?|(?:doughnut|donut)(?:\s+(?:chart|graph))?)\b"
)
_AGGREGATION_NOISE_RE = re.compile(
    r"\b(?:average|avg|mean|total|sum|combined|count|number\s+of|how\s+many|"
    r"minimum|min|maximum|max|median)\b"
)
_COMPARISON_NOISE_RE = re.compile(
    r"\b(?:highest|lowest|most|least|best|worst|greatest|smallest|top|bottom)\b"
)
_TIME_NOISE_RE = re.compile(
    r"\b(?:daily|weekly|monthly|quarterly|yearly|annually|"
    r"over\s+(?:time|day|week|month|quarter|year)|"
    r"by\s+(?:day|week|month|quarter|year)|(?:in|for)\s+\d{4}|"
    r"(?:last|this|next)\s+(?:day|week|month|quarter|year)|"
    r"between\s+\d{4}\s+and\s+\d{4})\b"
)
_EDGE_NOISE_RE = re.compile(
    r"^(?:(?:the|a|an|of|for|this)\s+)+|\s+(?:(?:the|a|an|of|for|this)\s*)+$"
)
_QUESTION_PREFIX_RE = re.compile(
    r"^(?:what\s+(?:is|was|are|were)(?:\s+the)?|how\s+many)\s+"
)
_QUESTION_SUFFIX_RE = re.compile(r"\s+(?:is|are|was|were)\s+there$")
_LEADING_LIMIT_RE = re.compile(
    rf"^(?P<limit>{_NUMBER_PATTERN})\s+(?P<term>.+)$"
)
_VAGUE_TERMS = {"this", "that", "something", "interesting", "stuff", "data", "it"}


def _intent_value(intent: QueryIntent | IntentClassification | None) -> QueryIntent | None:
    if intent is None:
        return None
    if isinstance(intent, IntentClassification):
        return intent.intent
    if isinstance(intent, str) and intent in {"VISUALIZATION", "INSIGHT", "UNKNOWN"}:
        return intent  # type: ignore[return-value]
    raise TypeError("intent must be a QueryIntent, IntentClassification, or None")


def _clean_term(phrase: str) -> str | None:
    term = normalize_text(phrase)
    term = _LEADING_NOISE_RE.sub("", term)
    term = _QUESTION_PREFIX_RE.sub("", term)
    term = _QUESTION_SUFFIX_RE.sub("", term)
    term = _CHART_NOISE_RE.sub(" ", term)
    term = _AGGREGATION_NOISE_RE.sub(" ", term)
    term = _COMPARISON_NOISE_RE.sub(" ", term)
    term = _TIME_NOISE_RE.sub(" ", term)
    term = " ".join(term.split())
    term = _EDGE_NOISE_RE.sub("", term).strip()
    if not term or term in _VAGUE_TERMS:
        return None
    if all(token in _VAGUE_TERMS for token in term.split()):
        return None
    return term


def _clean_measure_term(phrase: str) -> str | None:
    """Clean a measure while retaining qualifiers that disambiguate broad value terms."""
    normalized = normalize_text(phrase)
    qualified = re.search(r"\b(?:total|unit)\s+(?:value|price)\b", normalized)
    if qualified:
        return qualified.group(0)
    return _clean_term(phrase)


def _append_unique(values: list[str], value: str | None) -> None:
    if value and value not in values:
        values.append(value)


def _number_value(raw: str) -> int:
    return int(raw) if raw.isdecimal() else _NUMBER_WORDS[raw]


def _extract_time_terms(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for pattern in _TIME_PATTERNS:
        for match in pattern.finditer(text):
            found.append((match.start(), match.group(0)))
    result: list[str] = []
    for _position, expression in sorted(found, key=lambda item: item[0]):
        if expression not in result:
            result.append(expression)
    return result


def extract_entities(
    raw_text: str,
    intent: QueryIntent | IntentClassification | None = None,
) -> ExtractedEntities:
    """Extract analytical language concepts without consulting any dataset."""
    if not isinstance(raw_text, str):
        raise TypeError(f"raw_text must be a string, got {type(raw_text).__name__}")
    intent_value = _intent_value(intent)
    text = normalize_text(raw_text)
    if not text:
        return ExtractedEntities(signals=("empty_request",))

    signals: list[str] = []
    measures: list[str] = []
    dimensions: list[str] = []
    filters: list[QueryFilter] = []
    time_terms = _extract_time_terms(text)
    if time_terms:
        signals.append("time_expression")

    aggregation = normalize_aggregation(text)
    if aggregation != "NONE":
        signals.append("explicit_aggregation")

    comparison = normalize_comparison(text)
    limit: int | None = None
    top_bottom = _TOP_BOTTOM_RE.search(text)
    if top_bottom:
        limit = _number_value(top_bottom.group("limit"))
        comparison = "TOP_N" if top_bottom.group("direction") == "top" else "BOTTOM_N"
        _append_unique(dimensions, _clean_term(top_bottom.group("dimension")))
        _append_unique(measures, _clean_measure_term(top_bottom.group("measure")))
        signals.append("top_n_pattern" if comparison == "TOP_N" else "bottom_n_pattern")
        signals.append("by_pattern")
    elif comparison != "NONE":
        signals.append("comparison_language")

    explicit_chart = _EXPLICIT_CHART_RE.search(text)
    requested_chart = normalize_chart(explicit_chart.group(0)) if explicit_chart else None
    requested_chart_variant = normalize_chart_variant(text) if explicit_chart else None
    unsupported_chart_term: str | None = None
    if requested_chart is not None:
        signals.append("explicit_chart")
        if requested_chart_variant is not None:
            signals.append("explicit_chart_variant")
    else:
        chart_like = _ANY_CHART_RE.search(text)
        if chart_like is not None:
            unsupported_chart_term = chart_like.group(0)
            signals.append("unsupported_chart_term")

    # Filters are extracted before general relations so their right-hand text
    # does not become an invented dimension.
    analytical_text = _CHART_SUFFIX_RE.sub("", text).strip()
    where_filter = _WHERE_FILTER_RE.fullmatch(analytical_text)
    year_filter = _YEAR_FILTER_RE.fullmatch(analytical_text)
    named_filter = (
        _IN_NAMED_FILTER_RE.fullmatch(analytical_text)
        if not year_filter
        and not re.search(r"\bbrought\s+in\s+(?:the\s+)?(?:most|least)\b", analytical_text)
        else None
    )
    if where_filter:
        field = _clean_term(where_filter.group("field"))
        value = _clean_term(where_filter.group("value"))
        filters.append(QueryFilter(
            raw_text=f"{where_filter.group('field')} is {where_filter.group('value')}",
            column_term=field, operator="EQUALS", value_term=value,
        ))
        analytical_text = where_filter.group("prefix")
        signals.append("filter_pattern")
    elif year_filter:
        filters.append(QueryFilter(
            raw_text=f"{year_filter.group('prep')} {year_filter.group('year')}",
            operator="EQUALS", value_term=year_filter.group("year"),
        ))
        analytical_text = year_filter.group("prefix")
        signals.append("filter_pattern")
    elif named_filter:
        field = _clean_term(named_filter.group("field"))
        value = _clean_term(named_filter.group("value"))
        filters.append(QueryFilter(
            raw_text=f"in {named_filter.group('value')} {named_filter.group('field')}",
            column_term=field, operator="EQUALS", value_term=value,
        ))
        analytical_text = named_filter.group("prefix")
        signals.append("filter_pattern")

    if not top_bottom:
        natural_comparative = _NATURAL_COMPARATIVE_RE.fullmatch(analytical_text)
        comparative = _COMPARATIVE_QUESTION_RE.fullmatch(analytical_text)
        between = _BETWEEN_FIELDS_RE.search(analytical_text)
        two_field = _EXPLICIT_TWO_FIELD_RE.search(analytical_text)
        versus = _VERSUS_RE.search(analytical_text)
        relation = _RELATION_RE.search(analytical_text)

        if natural_comparative:
            direction = natural_comparative.group("leading") or natural_comparative.group("trailing")
            verb = natural_comparative.group("verb1") or natural_comparative.group("verb2")
            dimension_phrase = natural_comparative.group("dimension1") or natural_comparative.group("dimension2")
            comparison = "HIGHEST" if direction == "most" else "LOWEST"
            _append_unique(dimensions, _clean_term(dimension_phrase))
            if "order" in verb or "common" in verb:
                aggregation = "COUNT"
                signals.append("popularity_count_pattern")
            else:
                _append_unique(measures, "sales")
                signals.append("natural_sales_pattern")
            signals.append("natural_comparative_pattern")
        elif comparative:
            dimension_phrase = comparative.group("dimension")
            leading_limit = _LEADING_LIMIT_RE.fullmatch(dimension_phrase)
            if leading_limit:
                limit = _number_value(leading_limit.group("limit"))
                dimension_phrase = leading_limit.group("term")
                comparison = (
                    "TOP_N"
                    if comparison == "HIGHEST"
                    else "BOTTOM_N"
                )
                signals.append(
                    "top_n_pattern" if comparison == "TOP_N" else "bottom_n_pattern"
                )
            _append_unique(dimensions, _clean_term(dimension_phrase))
            _append_unique(measures, _clean_measure_term(comparative.group("measure") or ""))
            if comparative.group("comparison") == "smallest" and aggregation == "MIN":
                aggregation = "NONE"
            signals.append("comparative_question_pattern")
        elif between:
            _append_unique(measures, _clean_term(between.group("first")))
            _append_unique(measures, _clean_term(between.group("second")))
            signals.append("between_pattern")
        elif two_field:
            _append_unique(measures, _clean_term(two_field.group("first")))
            _append_unique(measures, _clean_term(two_field.group("second")))
            signals.append("two_field_pattern")
        elif versus:
            _append_unique(measures, _clean_term(versus.group("first")))
            _append_unique(measures, _clean_term(versus.group("second")))
            signals.append("versus_pattern")
        elif relation:
            _append_unique(measures, _clean_measure_term(relation.group("measure")))
            dimension = _clean_term(relation.group("dimension"))
            if relation.group("relation") == "over" and time_terms:
                pass
            else:
                _append_unique(dimensions, dimension)
            signals.append(f"{relation.group('relation')}_pattern")
        else:
            _append_unique(measures, _clean_measure_term(analytical_text))
            if measures:
                signals.append("single_term_pattern")

    if intent_value is not None:
        signals.append(f"intent_context:{intent_value.casefold()}")

    return ExtractedEntities(
        measure_terms=measures,
        dimension_terms=dimensions,
        aggregation=aggregation,
        comparison=comparison,
        requested_chart=requested_chart,
        requested_chart_variant=requested_chart_variant,
        unsupported_chart_term=unsupported_chart_term,
        filters=filters,
        time_terms=time_terms,
        limit=limit,
        signals=tuple(signals),
    )


__all__ = ["extract_entities"]
