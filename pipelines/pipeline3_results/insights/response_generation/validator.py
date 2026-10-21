"""Deterministic, high-risk factual validation for generated wording."""

from __future__ import annotations

import json
import re

from pipelines.pipeline3_results.insights.response_generation.schemas import (
    ModelResponseDraft,
    ResponseContext,
    ResponseValidation,
)


_NUMBER = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)
_CURRENCY_SYMBOL = re.compile(r"[$€£¥₹]")
_CAPITALIZED = re.compile(r"\b[A-Z][A-Za-z0-9_-]*\b")
_CAUSAL = re.compile(
    r"\b(?:because|caused?|causing|due\s+to|drives?|driven\s+by|"
    r"leads?\s+to|resulted?\s+in|responsible\s+for)\b",
    re.IGNORECASE,
)
_INTERNAL = re.compile(
    r"(?:\b(?:pandas|dataframe|verifiedresult|system\s+prompt|chain[- ]of[- ]thought|"
    r"stack\s+trace|plan_id|result_id|executor|python\s+code|sql\s+query)\b|```)",
    re.IGNORECASE,
)
_UNSUPPORTED_INFERENCE = re.compile(
    r"\b(?:correlat(?:e|ed|es|ion)|normal(?:ly)?\s+distributed|skew(?:ed|ness)?|outliers?)\b",
    re.IGNORECASE,
)
_LANGUAGE = re.compile(r"^(?:[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*|hinglish|und)$")
_SAFE_CAPITALIZED = {
    "A", "An", "And", "At", "Based", "Between", "By", "Count", "Distinct",
    "For", "From", "Highest", "In", "It", "Lowest", "Maximum", "Mean",
    "Median", "Minimum", "Most", "No", "Of", "On", "Range", "Records",
    "Sample", "The", "There", "This", "To", "Total", "Value", "Values",
}
_ENTITY_CLAIM = re.compile(
    r"(?P<entity>[\w][\w &/-]*?)\s+(?:has\s+the\s+(?:highest|lowest)|"
    r"are\s+tied\s+for|is\s+the\s+most\s+frequent|contributes?)\b",
    re.IGNORECASE,
)


def _numbers(value: str) -> set[str]:
    return {match.group(0) for match in _NUMBER.finditer(value)}


def _canonical_number(value: str) -> str:
    suffix = "%" if value.endswith("%") else ""
    return value.removesuffix("%").replace(",", "").lstrip("+") + suffix


def _allowed_entities(context: ResponseContext) -> set[str]:
    entities: set[str] = set()
    rows = context.verified_result.get("rows", [])
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            for value in row.values():
                if isinstance(value, str) and not _NUMBER.fullmatch(value):
                    entities.update(_CAPITALIZED.findall(value))
    source_text = " ".join([
        context.original_query,
        *context.semantic_labels.values(),
        *context.units_or_formats_if_known.values(),
    ])
    entities.update(_CAPITALIZED.findall(source_text))
    return entities


def _allowed_entity_values(context: ResponseContext) -> set[str]:
    entities: set[str] = set()
    rows = context.verified_result.get("rows", [])
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                entities.update(
                    value.casefold()
                    for value in row.values()
                    if isinstance(value, str) and not _NUMBER.fullmatch(value)
                )
    return entities


def validate_response(
    draft: ModelResponseDraft,
    context: ResponseContext,
) -> ResponseValidation:
    """Reject unsupported factual tokens without attempting prose understanding."""
    errors: list[str] = []
    text = draft.text.strip()
    if not text:
        errors.append("response is empty")
    if not _LANGUAGE.fullmatch(draft.language):
        errors.append("response language is invalid")
    elif (
        context.requested_language != "und"
        and draft.language.casefold() != context.requested_language.casefold()
    ):
        errors.append("response language does not match the requested language")
    if _CAUSAL.search(text):
        errors.append("unsupported causal language")
    if _INTERNAL.search(text):
        errors.append("internal implementation leakage")
    if _UNSUPPORTED_INFERENCE.search(text):
        errors.append("unsupported statistical inference")

    unit_text = json.dumps(
        context.units_or_formats_if_known, ensure_ascii=False, sort_keys=True
    )
    currency_codes = {
        "$": ("usd", "dollar"), "€": ("eur", "euro"),
        "£": ("gbp", "pound"), "¥": ("jpy", "cny", "yen", "yuan"),
        "₹": ("inr", "rupee"),
    }
    for symbol in set(_CURRENCY_SYMBOL.findall(text)):
        allowed = symbol in unit_text or any(
            code in unit_text.casefold() for code in currency_codes[symbol]
        )
        if not allowed:
            errors.append(f"unsupported currency unit: {symbol}")

    context_text = json.dumps(
        context.verified_result, ensure_ascii=False, sort_keys=True
    )
    allowed_numbers = {_canonical_number(item) for item in _numbers(context_text)}
    for number in _numbers(text):
        if _canonical_number(number) not in allowed_numbers:
            errors.append(f"unsupported numeric value: {number}")

    allowed_entities = _allowed_entities(context)
    for token in _CAPITALIZED.findall(text):
        if token not in allowed_entities and token not in _SAFE_CAPITALIZED:
            errors.append(f"unsupported factual entity: {token}")
    entity_values = _allowed_entity_values(context)
    for match in _ENTITY_CLAIM.finditer(text):
        claimed = match.group("entity").strip()
        candidates = [item.strip().casefold() for item in re.split(r"\s+and\s+", claimed)]
        if any(item not in entity_values for item in candidates):
            errors.append(f"unsupported factual entity claim: {claimed}")

    return ResponseValidation(valid=not errors, errors=errors)


__all__ = ["validate_response"]
