"""Evidence-aware, deterministic candidate generation for Pipeline 3 terms.

Scores are ordinal rule strengths, not calibrated probabilities:

* 1.00 exact normalized full name
* 0.86 exact token/subtoken match
* 0.76 direct Pipeline 1 semantic match
* 0.66 deterministic alias match
* +0.02 compatible Pipeline 3 role support (only after a semantic/name match)

This module deliberately returns candidate sets.  It never chooses a winner;
ambiguity decisions belong to the next NLP phase.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import pandas as pd

from pipelines.pipeline3_results.nlp.normalizer import normalize_text
from pipelines.pipeline3_results.nlp.schemas import (
    Aggregation,
    ColumnCandidate,
    ExtractedEntities,
    StructuredQuery,
    TermKind,
    TermResolution,
)
from pipelines.pipeline3_results.visualization.column_classifier import (
    _extract_column_evidence,
)
from pipelines.pipeline3_results.visualization.schemas import DatasetClassification


EXACT_NAME_STRENGTH = 1.00
TOKEN_STRENGTH = 0.86
SEMANTIC_STRENGTH = 0.76
ALIAS_STRENGTH = 0.66
ROLE_SUPPORT_BONUS = 0.02

_GENERIC_TOKENS = {
    "a", "an", "the", "id", "no", "number", "value", "type", "data",
    "field", "column", "code",
}

# Concept families identify possible semantics; they never prescribe a column.
_ALIASES: dict[str, frozenset[str]] = {
    "sales": frozenset({
        "sale", "sales", "sell", "selling", "sold", "revenue", "income",
        "turnover", "earnings", "money", "proceeds", "price",
    }),
    "revenue": frozenset({
        "sale", "sales", "sell", "selling", "sold", "revenue", "income",
        "turnover", "earnings", "money", "proceeds", "price",
    }),
    "value": frozenset({"value", "amount", "monetary"}),
    "category": frozenset({"category", "type", "group", "segment"}),
    "income": frozenset({
        "income", "earnings", "revenue", "salary", "wage", "pay",
    }),
    "salary": frozenset({
        "salary", "wage", "wages", "pay", "compensation", "earnings",
    }),
    "department": frozenset({"department", "dept", "division", "team"}),
    "region": frozenset({"region", "territory", "area", "zone"}),
    "time": frozenset({
        "time", "date", "datetime", "timestamp", "created", "updated", "year",
        "month", "day",
    }),
    "date": frozenset({"date", "datetime", "timestamp", "created", "updated"}),
    "month": frozenset({"month", "date", "datetime", "timestamp"}),
    "year": frozenset({"year", "date", "datetime", "timestamp"}),
    "product": frozenset({"product", "item", "sku", "merchandise"}),
    "customer": frozenset({"customer", "client", "buyer", "consumer"}),
    "discount": frozenset({"discount", "rebate", "markdown"}),
    "rating": frozenset({"rating", "score", "rank", "evaluation"}),
    "quantity": frozenset({"quantity", "count", "units", "items", "volume"}),
}

_SEMANTIC_TEXT_FIELDS = (
    "concept_description",
    "semantic_description",
    "column_description",
    "description",
    "reasoning_summary",
    "semantic_reason",
    "reason",
    "semantic_type",
    "semantic_role",
    "value_function",
    "numeric_meaning",
    "role",
    "detected_patterns",
)

_ROLE_COMPATIBILITY: dict[TermKind, frozenset[str]] = {
    "MEASURE": frozenset({"MEASURE"}),
    "RELATIONSHIP": frozenset({"MEASURE"}),
    "DIMENSION": frozenset({"CATEGORY", "TEMPORAL", "BOOLEAN"}),
    "FILTER": frozenset({"CATEGORY", "TEMPORAL", "BOOLEAN"}),
    "TIME": frozenset({"TEMPORAL"}),
}


def _tokens(value: str) -> list[str]:
    return re.findall(r"\b\w+\b", normalize_text(value))


def _singular(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _meaningful_forms(value: str) -> set[str]:
    forms: set[str] = set()
    for token in _tokens(value):
        if token in _GENERIC_TOKENS or len(token) < 3:
            continue
        forms.add(_singular(token))
    return forms


def _canonical_time_term(term: str) -> str:
    normalized = normalize_text(term)
    replacements = {
        "monthly": "month", "yearly": "year", "annually": "year",
        "daily": "day", "weekly": "week", "quarterly": "quarter",
    }
    if normalized in replacements:
        return replacements[normalized]
    normalized = re.sub(r"^(?:over|by|in|for|last|this|next)\s+", "", normalized)
    return normalized


def _alias_family(term: str) -> tuple[str | None, frozenset[str]]:
    # Known concept words remain meaningful even when deliberately excluded
    # from generic token matching (for example, broad "value" and "type").
    forms = {_singular(token) for token in _tokens(term)}
    for concept, aliases in _ALIASES.items():
        if concept in forms or forms.intersection(aliases):
            return concept, aliases
    return None, frozenset()


def _conceptual_candidate(
    term: str,
    kind: TermKind,
    column_forms: set[str],
    evidence: Mapping[str, Any],
    role: str | None,
    aggregation: Aggregation,
) -> tuple[float, list[str], dict[str, Any]]:
    """Return evidence-backed candidates for ordinary analytical concepts.

    These rules produce candidate sets only.  Strong total/revenue evidence may
    outrank unit-level monetary evidence, while broad value language deliberately
    leaves all supported measure-like meanings tied for clarification.
    """
    concept, _aliases = _alias_family(term)
    semantic_fields = _semantic_fields(evidence)
    semantic_text = " ".join(text for _key, text in semantic_fields)
    semantic_forms = _meaningful_forms(semantic_text)
    all_forms = set(column_forms) | semantic_forms
    snippets = {
        key: _json_scalar(text[:240])
        for key, text in semantic_fields
        if key in {"concept_description", "semantic_description", "reasoning_summary", "value_function", "numeric_meaning"}
    }

    if aggregation == "COUNT" and kind == "MEASURE" and role == "IDENTIFIER":
        requested = {_singular(token) for token in _tokens(term)}
        if requested and requested.intersection(column_forms):
            return 0.94, ["semantic_count_identifier_match"], snippets

    if kind == "DIMENSION":
        normalized_term = normalize_text(term)
        normalized_semantics = normalize_text(semantic_text)
        entity_name_pattern = re.compile(
            rf"\bname\s+of\s+(?:the\s+)?{re.escape(normalized_term)}\b"
        )
        if (
            normalized_term
            and role == "CATEGORY"
            and entity_name_pattern.search(normalized_semantics)
        ):
            return 0.94, ["semantic_entity_name_match"], snippets

    if concept in {"sales", "revenue"} and kind in {"MEASURE", "RELATIONSHIP"}:
        monetary = {
            "price", "amount", "monetary", "revenue", "sale", "income",
            "turnover", "earning", "proceed",
        }
        if role != "MEASURE" or not all_forms.intersection(monetary):
            return 0.0, [], {}
        total_level = {
            "total", "revenue", "sale", "income", "turnover", "earning", "proceed",
        }
        if all_forms.intersection(total_level):
            return SEMANTIC_STRENGTH, ["semantic_concept_match:sales"], snippets
        return ALIAS_STRENGTH, ["semantic_concept_candidate:sales"], snippets

    if concept == "value" and kind in {"MEASURE", "RELATIONSHIP"}:
        value_indicators = {
            "value", "amount", "price", "revenue", "sale", "income", "cost",
            "quantity", "unit", "volume", "score", "measurement",
        }
        if role == "MEASURE" and all_forms.intersection(value_indicators):
            # A qualifier such as "total value" may safely separate total-level
            # fields from equally plausible unit/quantity measures.
            requested = set(_tokens(term))
            if "total" in requested and "total" in all_forms:
                return SEMANTIC_STRENGTH, ["semantic_concept_match:total_value"], snippets
            if "unit" in requested and "unit" in all_forms:
                return SEMANTIC_STRENGTH, ["semantic_concept_match:unit_value"], snippets
            return ALIAS_STRENGTH, ["semantic_concept_candidate:value"], snippets

    return 0.0, [], {}


def _json_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ", ".join(str(item) for item in value[:10])
    return str(value)


def _semantic_fields(evidence: Mapping[str, Any]) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for key in _SEMANTIC_TEXT_FIELDS:
        value = evidence.get(key)
        if value is None:
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            text = " ".join(str(item) for item in value[:20])
        elif isinstance(value, (str, int, float, bool)):
            text = str(value)
        else:
            continue
        if text.strip():
            fields.append((key, text))
    return fields


def _semantic_match(
    term: str,
    evidence: Mapping[str, Any],
) -> tuple[float, list[str], dict[str, Any]]:
    term_forms = _meaningful_forms(term)
    concept, aliases = _alias_family(term)
    best = 0.0
    sources: list[str] = []
    snippets: dict[str, Any] = {}
    for key, text in _semantic_fields(evidence):
        semantic_forms = _meaningful_forms(text)
        direct = bool(term_forms) and term_forms.issubset(semantic_forms)
        alias = bool(aliases) and bool(aliases.intersection(semantic_forms))
        if direct:
            best = max(best, SEMANTIC_STRENGTH)
            if key in {"concept_description", "semantic_description", "column_description", "description", "reasoning_summary"}:
                source = "semantic_description_match"
            elif key in {"semantic_role", "role"}:
                source = "semantic_role_match"
            else:
                source = "semantic_metadata_match"
            if source not in sources:
                sources.append(source)
            snippets[key] = _json_scalar(text[:240])
        elif alias:
            best = max(best, ALIAS_STRENGTH)
            source = f"semantic_alias_match:{concept}"
            if source not in sources:
                sources.append(source)
            snippets[key] = _json_scalar(text[:240])
    return best, sources, snippets


def _role_for(
    classification: DatasetClassification | None,
    column_name: str,
) -> str | None:
    if classification is None:
        return None
    classified = classification.get_column(column_name)
    return classified.role if classified is not None else None


def _resolve_term(
    term: str,
    kind: TermKind,
    columns: list[str],
    pipeline1_report: Any,
    classification: DatasetClassification | None,
    aggregation: Aggregation,
) -> TermResolution:
    match_term = _canonical_time_term(term) if kind == "TIME" else normalize_text(term)
    term_forms = _meaningful_forms(match_term)
    alias_concept, aliases = _alias_family(match_term)
    candidates: list[tuple[int, ColumnCandidate]] = []

    for index, column in enumerate(columns):
        normalized_column = normalize_text(column)
        column_forms = _meaningful_forms(column)
        sources: list[str] = []
        strength = 0.0

        if match_term and normalized_column == match_term:
            strength = EXACT_NAME_STRENGTH
            sources.append("exact_normalized_name")
        elif term_forms and term_forms.issubset(column_forms):
            strength = TOKEN_STRENGTH
            sources.append("exact_token_match")
        elif aliases and aliases.intersection(column_forms):
            strength = ALIAS_STRENGTH
            sources.append(f"alias_match:{alias_concept}")

        raw_evidence = _extract_column_evidence(pipeline1_report, column)
        semantic_strength, semantic_sources, semantic_snippets = _semantic_match(
            match_term, raw_evidence
        )
        strength = max(strength, semantic_strength)
        for source in semantic_sources:
            if source not in sources:
                sources.append(source)

        role = _role_for(classification, column)
        concept_strength, concept_sources, concept_snippets = _conceptual_candidate(
            match_term, kind, column_forms, raw_evidence, role, aggregation
        )
        strength = max(strength, concept_strength)
        for source in concept_sources:
            if source not in sources:
                sources.append(source)
        semantic_snippets.update(concept_snippets)

        # No name, alias, or semantic evidence means role compatibility cannot
        # manufacture a candidate.
        if strength == 0.0:
            continue

        if role in _ROLE_COMPATIBILITY[kind]:
            sources.append(f"analytical_role_support:{role}")
            strength = min(1.0, strength + ROLE_SUPPORT_BONUS)

        relevant_semantics: dict[str, Any] = dict(semantic_snippets)
        for key in ("semantic_type", "semantic_role", "value_function", "numeric_meaning"):
            if key in raw_evidence and (semantic_sources or key in semantic_snippets):
                relevant_semantics[key] = _json_scalar(raw_evidence[key])

        candidates.append((index, ColumnCandidate(
            column_name=column,
            score=round(strength, 2),
            evidence_sources=sources,
            semantic_evidence=relevant_semantics,
            analytical_role=role,
        )))

    candidates.sort(key=lambda pair: (
        -(pair[1].score or 0.0), pair[0], pair[1].column_name.casefold(),
    ))
    ordered = [candidate for _index, candidate in candidates]
    return TermResolution(
        term=term,
        kind=kind,
        candidates=ordered,
        resolved_column=None,
        confidence=None,
        reason=(
            f"{len(ordered)} evidence-supported candidate(s); selection deferred"
            if ordered else "no evidence-supported candidates"
        ),
    )


def _terms_from_input(
    query: ExtractedEntities | StructuredQuery,
) -> list[tuple[str, TermKind]]:
    terms: list[tuple[str, TermKind]] = []
    relationship = isinstance(query, ExtractedEntities) and any(
        signal in {"between_pattern", "versus_pattern", "two_field_pattern"}
        for signal in query.signals
    )
    measure_kind: TermKind = "RELATIONSHIP" if relationship else "MEASURE"
    for term in query.measure_terms:
        terms.append((term, measure_kind))
    for term in query.dimension_terms:
        terms.append((term, "DIMENSION"))
    for term in query.time_terms:
        terms.append((term, "TIME"))
    for item in query.filters:
        if item.column_term:
            terms.append((item.column_term, "FILTER"))

    unique: list[tuple[str, TermKind]] = []
    seen: set[tuple[str, TermKind]] = set()
    for term, kind in terms:
        key = (normalize_text(term), kind)
        if key not in seen:
            seen.add(key)
            unique.append((term, kind))
    return unique


def resolve_columns(
    query: ExtractedEntities | StructuredQuery,
    dataframe: pd.DataFrame,
    pipeline1_report: Any = None,
    classification: DatasetClassification | None = None,
) -> list[TermResolution]:
    """Return evidence-bearing candidate sets for every unresolved query term."""
    if not isinstance(query, (ExtractedEntities, StructuredQuery)):
        raise TypeError("query must be ExtractedEntities or StructuredQuery")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"dataframe must be a pandas DataFrame, got {type(dataframe).__name__}")
    if classification is not None and not isinstance(classification, DatasetClassification):
        raise TypeError("classification must be a DatasetClassification or None")

    columns = [str(column) for column in dataframe.columns]
    return [
        _resolve_term(
            term, kind, columns, pipeline1_report, classification,
            query.aggregation,
        )
        for term, kind in _terms_from_input(query)
    ]


resolve_column_terms = resolve_columns


__all__ = [
    "ALIAS_STRENGTH",
    "EXACT_NAME_STRENGTH",
    "ROLE_SUPPORT_BONUS",
    "SEMANTIC_STRENGTH",
    "TOKEN_STRENGTH",
    "resolve_column_terms",
    "resolve_columns",
]
