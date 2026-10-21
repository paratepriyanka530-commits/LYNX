"""Safe provider wording and deterministic fallback for summary facts."""

from __future__ import annotations

import json
import re
from typing import Any

from intelligence.llm.provider import LLMProvider
from pipelines.pipeline3_results.insights.landing.schemas import (
    NaturalLanguageSummary,
    SummaryFact,
    SummaryModelDraft,
    SummaryPackage,
)
from pipelines.pipeline3_results.insights.response_generation.context_builder import (
    format_number,
)


SUMMARY_SYSTEM_PROMPT = """You are LYNX's dataset summary explanation layer.
Use only the supplied verified summary facts.
Do not calculate, invent values, trends, causes, entities, or business conclusions.
Mention supplied caveats when important.
Keep the summary concise and prefer three to five useful statements.
Respond in the requested language where possible.
Do not expose implementation details or chain-of-thought.
Return JSON with exactly two fields: text and language."""

_NUMBER = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)
_CAUSAL_OR_CONCLUSION = re.compile(
    r"\b(?:because|caused?|due\s+to|drives?|leads?\s+to|strong\s+demand|"
    r"performing\s+well|customers?\s+prefer|indicates?|proves?|growing|declining)\b",
    re.IGNORECASE,
)
_INTERNAL = re.compile(
    r"\b(?:dataframe|pandas|executor|verifiedresult|plan_id|result_id|system prompt|"
    r"chain-of-thought)\b|```",
    re.IGNORECASE,
)
_CAPITALIZED = re.compile(r"\b[A-Z][A-Za-z0-9_-]*\b")
_COMPARATIVE = re.compile(
    r"\b(?:highest|lowest|tied|higher\s+than|lower\s+than|more\s+than|less\s+than)\b",
    re.IGNORECASE,
)
_GROUP_CLAIM = re.compile(
    r"(?P<entity>[\w][\w &/-]*?)\s+(?:has\s+the\s+(?:highest|lowest)|"
    r"are\s+tied\s+for)\b",
    re.IGNORECASE,
)
_SAFE_WORDS = {
    "A", "An", "And", "Average", "Category", "Columns", "Dataset", "Distinct",
    "Each", "Fields", "For", "From", "It", "Measures", "No", "Of", "Records",
    "Relationships", "Rows", "The", "This", "Time", "Values", "With",
}


def _display(value: Any, key: str = "") -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        rendered = format_number(value)
        return f"{rendered}%" if "percentage" in key else rendered
    if isinstance(value, list):
        return [_display(item, key) for item in value]
    if isinstance(value, dict):
        return {name: _display(item, name) for name, item in value.items()}
    return value


def serialize_summary(package: SummaryPackage) -> dict[str, Any]:
    return {
        "status": package.status,
        "facts": [{
            "fact_type": fact.fact_type,
            "values": _display(fact.values),
            "caveats": fact.caveats,
        } for fact in package.facts],
        "opportunities": package.opportunities,
        "warnings": package.warnings,
    }


def _canonical_number(value: str) -> str:
    suffix = "%" if value.endswith("%") else ""
    return value.removesuffix("%").replace(",", "").lstrip("+") + suffix


def _validate_draft(
    draft: SummaryModelDraft,
    serialized: dict[str, Any],
    requested_language: str,
) -> bool:
    if draft.language.casefold() != requested_language.casefold():
        return False
    if _CAUSAL_OR_CONCLUSION.search(draft.text) or _INTERNAL.search(draft.text):
        return False
    ranking_facts = [
        item for item in serialized["facts"]
        if item.get("fact_type") == "GROUP_RANKING"
    ]
    if _COMPARATIVE.search(draft.text) and not ranking_facts:
        return False
    facts_text = json.dumps(serialized, ensure_ascii=False, sort_keys=True)
    allowed_numbers = {
        _canonical_number(match.group(0)) for match in _NUMBER.finditer(facts_text)
    }
    if any(
        _canonical_number(match.group(0)) not in allowed_numbers
        for match in _NUMBER.finditer(draft.text)
    ):
        return False
    allowed_words = set(_CAPITALIZED.findall(facts_text))
    if not all(
        token in allowed_words or token in _SAFE_WORDS
        for token in _CAPITALIZED.findall(draft.text)
    ):
        return False
    allowed_leaders = {
        str(leader).casefold()
        for fact in ranking_facts
        for leader in fact.get("values", {}).get("leaders", [])
    }
    for match in _GROUP_CLAIM.finditer(draft.text):
        claimed = [
            item.strip().casefold()
            for item in re.split(r"\s+and\s+", match.group("entity"))
        ]
        if any(item not in allowed_leaders for item in claimed):
            return False
    return True


def _fact_sentence(fact: SummaryFact) -> str:
    values = fact.values
    if fact.fact_type == "DATASET_OVERVIEW":
        return (
            f"The dataset contains {int(values['row_count']):,} records across "
            f"{int(values['column_count']):,} columns."
        )
    if fact.fact_type == "DATA_QUALITY":
        if "columns_with_missing_values" in values:
            return (
                f"Reported metadata shows {int(values['columns_with_missing_values']):,} "
                "columns with missing values."
            )
        if "quality_score" in values:
            return f"The reported data-quality score is {format_number(values['quality_score'])}."  # type: ignore[arg-type]
        return f"The cleaning report records {int(values['reported_issue_count']):,} quality issues."
    if fact.fact_type == "SEMANTIC_STRUCTURE":
        parts = []
        if values.get("measures"):
            parts.append("measures such as " + ", ".join(values["measures"]))  # type: ignore[arg-type]
        if values.get("dimensions"):
            parts.append("dimensions such as " + ", ".join(values["dimensions"]))  # type: ignore[arg-type]
        if values.get("temporal_fields"):
            parts.append("time fields such as " + ", ".join(values["temporal_fields"]))  # type: ignore[arg-type]
        return "The dataset includes " + "; ".join(parts) + "."
    if fact.fact_type == "MEASURE_AGGREGATE":
        return f"The average {values['label']} is {format_number(values['value'])}."  # type: ignore[arg-type]
    if fact.fact_type == "GROUP_RANKING":
        leaders = [str(item) for item in values["leaders"]]  # type: ignore[union-attr]
        count = format_number(values["value"])  # type: ignore[arg-type]
        if values.get("tie"):
            return f"{' and '.join(leaders)} are tied for the highest {values['label']} record count at {count}."
        return f"{leaders[0]} has the highest {values['label']} record count at {count}."
    if fact.fact_type == "ENTITY_COUNT":
        return f"There are {format_number(values['value'])} distinct {values['label']} values."  # type: ignore[arg-type]
    if fact.fact_type == "NUMERIC_RANGE":
        return f"The {values['label']} range is {format_number(values['value'])}."  # type: ignore[arg-type]
    if fact.fact_type == "TREND_OPPORTUNITY":
        return f"Time-based trend analysis is available using {values['label']}."
    if fact.fact_type == "RELATIONSHIP_OPPORTUNITY":
        return f"Relationships between {values['first_label']} and {values['second_label']} can be explored."
    return f"Comparisons across {values['label']} are available."


def deterministic_summary(
    package: SummaryPackage, *, requested_language: str = "en"
) -> NaturalLanguageSummary:
    text = " ".join(_fact_sentence(fact) for fact in package.facts)
    if not text:
        text = "No reliable dataset summary facts are currently available."
    return NaturalLanguageSummary(
        text=text,
        language="en",
        status="FALLBACK",
        fallback_used=True,
        caveats=[*package.warnings, *[item for fact in package.facts for item in fact.caveats]],
    )


class SummaryWordingService:
    def generate(
        self,
        package: SummaryPackage,
        provider: LLMProvider | None,
        *,
        requested_language: str = "en",
    ) -> NaturalLanguageSummary:
        if provider is None:
            return deterministic_summary(package, requested_language=requested_language)
        serialized = serialize_summary(package)
        prompt = (
            f"{SUMMARY_SYSTEM_PROMPT}\n\nRequested language: {requested_language}\n"
            f"Verified summary facts:\n{json.dumps(serialized, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
        )
        try:
            raw = provider.generate_structured(prompt, SummaryModelDraft)
            draft = SummaryModelDraft.model_validate(raw)
            if not _validate_draft(draft, serialized, requested_language):
                return deterministic_summary(package, requested_language=requested_language)
            return NaturalLanguageSummary(
                text=draft.text,
                language=draft.language,
                status="READY",
                fallback_used=False,
                caveats=[*package.warnings, *[item for fact in package.facts for item in fact.caveats]],
            )
        except Exception:
            return deterministic_summary(package, requested_language=requested_language)


__all__ = [
    "SUMMARY_SYSTEM_PROMPT", "SummaryWordingService", "deterministic_summary",
    "serialize_summary",
]
