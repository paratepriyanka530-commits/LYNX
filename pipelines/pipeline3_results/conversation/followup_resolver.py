"""Deterministic recognition and structured merging of safe follow-ups."""

from __future__ import annotations

import re

from pipelines.pipeline3_results.conversation.schemas import (
    ConversationQueryState,
    FollowupResolution,
)
from pipelines.pipeline3_results.nlp.normalizer import (
    normalize_aggregation,
    normalize_chart,
    normalize_chart_variant,
    normalize_text,
)
from pipelines.pipeline3_results.nlp.schemas import QueryFilter


_TOP_BOTTOM = re.compile(
    r"^(?:show\s+)?(?P<direction>top|bottom)\s+(?P<limit>\d+)\s+(?:instead)?$"
)
_DIMENSION = re.compile(r"^(?:by|group\s+by)\s+(?P<term>.+?)(?:\s+instead)?$")
_MEASURE = re.compile(r"^use\s+(?P<term>.+?)\s+instead$")
_ONLY_FOR_YEAR = re.compile(r"^(?:only|just)?\s*(?:for|in)\s+(?P<year>\d{4})$")
_LAST_YEAR = re.compile(r"^(?:only\s+|just\s+)?last\s+year$")
_ONLY_VALUE = re.compile(r"^(?:only|just)\s+(?P<value>.+)$")


def resolve_followup(
    user_text: str,
    previous: ConversationQueryState | None,
) -> FollowupResolution:
    """Return a field-level modification, never a concatenated message history."""
    if previous is None:
        return FollowupResolution(is_followup=False)

    text = normalize_text(user_text)
    update: dict[str, object]

    top_bottom = _TOP_BOTTOM.fullmatch(text)
    if top_bottom:
        update = {
            "comparison": (
                "TOP_N" if top_bottom.group("direction") == "top" else "BOTTOM_N"
            ),
            "limit": int(top_bottom.group("limit")),
        }
        return _result(previous, "top_bottom", update)

    aggregation = normalize_aggregation(text)
    if aggregation != "NONE" and (
        "instead" in text or text.startswith(("use ", "average", "mean", "total", "sum"))
    ):
        return _result(previous, "aggregation", {"aggregation": aggregation})

    chart = normalize_chart(text)
    if chart is not None and (
        "instead" in text or text.startswith(("make ", "use ", "change "))
    ):
        return _result(previous, "chart", {
            "requested_chart": chart,
            "requested_chart_variant": normalize_chart_variant(text),
        })

    if re.fullmatch(r"(?:what\s+about\s+the\s+)?lowest(?:\s+instead)?", text):
        return _result(previous, "comparison", {"comparison": "LOWEST"})
    if re.fullmatch(r"(?:what\s+about\s+the\s+)?highest(?:\s+instead)?", text):
        return _result(previous, "comparison", {"comparison": "HIGHEST"})

    year = _ONLY_FOR_YEAR.fullmatch(text)
    if year:
        return _result(
            previous, "temporal", {"temporal_terms": (f"in {year.group('year')}",)}
        )
    if _LAST_YEAR.fullmatch(text):
        return _result(previous, "temporal", {"temporal_terms": ("last year",)})

    dimension = _DIMENSION.fullmatch(text)
    if dimension:
        return _result(previous, "dimension", {
            "dimension": None,
            "dimension_term": dimension.group("term"),
        })

    measure = _MEASURE.fullmatch(text)
    if measure:
        return _result(previous, "measure", {
            "measure": None,
            "measure_term": measure.group("term"),
        })

    only = _ONLY_VALUE.fullmatch(text)
    if only and previous.dimension is not None:
        query_filter = QueryFilter(
            raw_text=user_text,
            column_term=previous.dimension,
            resolved_column=previous.dimension,
            operator="EQUALS",
            value_term=only.group("value"),
        )
        return _result(previous, "filter", {"filters": (*previous.filters, query_filter)})

    return FollowupResolution(is_followup=False)


def _result(
    previous: ConversationQueryState,
    modification: str,
    update: dict[str, object],
) -> FollowupResolution:
    query = previous.model_copy(update=update)
    return FollowupResolution(
        is_followup=True,
        modification=modification,
        query=query,
        signals=(f"conversation:followup:{modification}", "conversation:structured_merge"),
    )


def compile_structured_request(query: ConversationQueryState) -> str:
    """Compile one structured state into a canonical single-turn request."""
    measure = query.measure or query.measure_term
    dimension = query.dimension or query.dimension_term
    aggregation_words = {
        "SUM": "total", "MEAN": "average", "COUNT": "count",
        "MIN": "minimum", "MAX": "maximum", "MEDIAN": "median", "NONE": "",
    }
    aggregate = aggregation_words[query.aggregation]
    measure_phrase = " ".join(part for part in (aggregate, measure) if part)

    if query.intent == "VISUALIZATION":
        chart_words = {
            "BAR": "bar chart of", "LINE": "line chart of",
            "SCATTER": "scatter plot of", "HISTOGRAM": "histogram of",
            "COUNT_BAR": "count chart of",
            "PIE": (
                "doughnut chart of"
                if query.requested_chart_variant == "DOUGHNUT"
                else "pie chart of"
            ),
        }
        prefix = chart_words.get(query.requested_chart, "show")
        parts = [prefix, measure_phrase]
        if dimension:
            parts.extend(["by", dimension])
        request = " ".join(part for part in parts if part)
    elif query.comparison in {"TOP_N", "BOTTOM_N"}:
        direction = "top" if query.comparison == "TOP_N" else "bottom"
        request = f"What are the {direction} {query.limit or 5} {dimension} by {measure_phrase}?"
    elif query.comparison in {"HIGHEST", "LOWEST"}:
        direction = "highest" if query.comparison == "HIGHEST" else "lowest"
        request = f"Which {dimension} has the {direction} {measure_phrase}?"
    else:
        request = f"What is the {measure_phrase}?"

    for query_filter in query.filters:
        if query_filter.column_term and query_filter.value_term:
            request += f" where {query_filter.column_term} is {query_filter.value_term}"
    for temporal in query.temporal_terms:
        request += f" {temporal}"
    return " ".join(request.split())


__all__ = ["compile_structured_request", "resolve_followup"]
