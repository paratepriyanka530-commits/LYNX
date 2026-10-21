"""Deterministic and optional provider wording over selected XAI facts."""

from __future__ import annotations

import json
import re
from typing import Any

from intelligence.llm.provider import LLMProvider
from pipelines.pipeline3_results.visualization.intelligence.schemas import (
    ChartFamily, VisualizationPlan,
)
from .schemas import XAIFact, XAIModelDraft


XAI_SYSTEM_PROMPT = """You are LYNX's chart explanation wording layer.
Use only the supplied deterministic XAI facts, rationale, labels, and limitations.
Never calculate or invent numbers, percentages, entities, trends, correlations, outliers, causes, statistical significance, or business conclusions.
Keep the chart rationale to one short sentence and findings to one to three short sentences.
Keep observations separate from limitations and do not expose implementation details.
Return JSON with exactly: why_this_chart, what_it_shows, limitations, language."""

_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_CAPITAL = re.compile(r"\b[A-Z][A-Za-z0-9_-]*\b")
_CAUSAL = re.compile(r"\b(?:because|caused?|causing|due\s+to|drives?|driven\s+by|leads?\s+to|resulted?\s+in|responsible\s+for)\b", re.I)
_SIGNIFICANCE = re.compile(r"\b(?:statistically\s+significant|p[- ]?value|confidence\s+interval)\b", re.I)
_BUSINESS = re.compile(r"\b(?:customers?\s+prefer|strong\s+demand|performing\s+well|business\s+should|recommended\s+action|market\s+success)\b", re.I)
_INTERNAL = re.compile(r"\b(?:dataframe|pandas|executor|verifiedresult|plan_id|result_id|system\s+prompt|chain[- ]of[- ]thought|python|sql)\b|```", re.I)
_SAFE = {"A", "An", "And", "Area", "Across", "Bar", "Based", "Box", "Chart", "Correlation", "Count", "Distribution", "Doughnut", "From", "Heatmap", "Histogram", "It", "Line", "No", "Only", "Pearson", "Pie", "Scatter", "The", "This", "To", "Value", "Values"}


def _fmt(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


def chart_rationale(family: ChartFamily) -> str:
    return {
        ChartFamily.BAR: "Bar charts are suitable for comparing values across discrete categories.",
        ChartFamily.GROUPED_BAR: "Grouped bar charts are suitable for comparing series across discrete categories.",
        ChartFamily.STACKED_BAR: "Stacked bar charts are suitable for comparing additive totals and their components.",
        ChartFamily.LINE: "Line charts are suitable for showing ordered change over time.",
        ChartFamily.AREA: "Area charts are suitable for showing ordered change over time while emphasizing magnitude.",
        ChartFamily.PIE: "Pie charts show how a small number of non-negative categories contribute to a whole.",
        ChartFamily.DOUGHNUT: "Doughnut charts show how a small number of non-negative categories contribute to a whole.",
        ChartFamily.SCATTER: "Scatter plots are suitable for examining the relationship between two numeric variables.",
        ChartFamily.HISTOGRAM: "Histograms show how numeric values are distributed across ranges.",
        ChartFamily.BOX_PLOT: "Box plots summarize the median, quartiles, spread, and statistical outliers of numeric values.",
        ChartFamily.HEATMAP: "Heatmaps are suitable for comparing values across two categorical dimensions.",
    }.get(family, "A verified explanation is not yet available for this chart family.")


def selection_rationale(plan: VisualizationPlan) -> str:
    purpose = plan.purpose.value.replace("_", " ").lower()
    if plan.source == "USER_REQUESTED":
        return f"This view was produced from the user's requested {purpose} analysis."
    return f"LYNX selected this verified view to provide a {purpose} perspective in the dashboard."


def _sentence(fact: XAIFact) -> str | None:
    value = fact.values
    kind = fact.fact_type
    label = str(value.get("value_label", "Value"))
    if kind == "MAXIMUM":
        where = value.get("category", value.get("period"))
        if value.get("series") is not None:
            where = f"{where} / {value['series']}"
        return f"The highest {label} is {_fmt(value['value'])} at {where}."
    if kind == "MINIMUM":
        where = value.get("category", value.get("period"))
        return f"The lowest {label} is {_fmt(value['value'])} at {where}."
    if kind == "ABSOLUTE_CHANGE" and "first_value" in value:
        return f"{label} {('increased' if value['change'] > 0 else 'decreased' if value['change'] < 0 else 'was unchanged')} from {_fmt(value['first_value'])} at {value['first_period']} to {_fmt(value['last_value'])} at {value['last_period']}."
    if kind == "ABSOLUTE_CHANGE":
        return f"The displayed difference is {_fmt(value['difference'])}."
    if kind == "PERCENT_CHANGE":
        return f"The verified percentage change is {_fmt(value['percent'])}%."
    if kind == "TOP_SHARE":
        return f"{value['category']} contributes {_fmt(value['share_percent'])}% of the displayed total."
    if kind == "TIE":
        tied = value.get("highest") or value.get("lowest") or []
        return f"{' and '.join(str(item) for item in tied)} are tied at the displayed extreme."
    if kind == "CORRELATION":
        return f"Across {_fmt(value['pair_count'])} valid pairs, Pearson correlation is {_fmt(value['coefficient'])}, a {value['strength']} {value['direction']} relationship."
    if kind == "PAIR_COUNT":
        return f"The chart contains {_fmt(value['pair_count'])} valid pairs."
    if kind == "DOMINANT_BIN":
        return f"The most populated range is {_fmt(value['bin_start'])} to {_fmt(value['bin_end'])}, with {_fmt(value['count'])} values."
    if kind == "RANGE":
        if "minimum" in value:
            return f"The represented range spans {_fmt(value['minimum'])} to {_fmt(value['maximum'])}."
    if kind == "MEDIAN":
        if "median" in value:
            group = f" for {value['group']}" if value.get("group") is not None else ""
            return f"The median{group} is {_fmt(value['median'])}."
    if kind == "IQR":
        group = f" for {value['group']}" if value.get("group") is not None else ""
        return f"The interquartile range{group} is {_fmt(value['iqr'])}, from {_fmt(value['q1'])} to {_fmt(value['q3'])}."
    if kind == "OUTLIER_COUNT":
        return f"The box-plot rule identifies {_fmt(value['count'])} statistical outliers."
    if kind == "HIGH_VALUE_CELL":
        return f"The highest cell is {value['x']} / {value['y']} at {_fmt(value['value'])}."
    if kind == "LOW_VALUE_CELL":
        return f"The lowest cell is {value['x']} / {value['y']} at {_fmt(value['value'])}."
    return None


def deterministic_findings(facts: list[XAIFact]) -> str:
    sentences = [sentence for fact in facts if (sentence := _sentence(fact))]
    return " ".join(sentences[:3]) or "No supported finding could be derived from the verified chart result."


def serialize_for_provider(
    facts: list[XAIFact], rationale: str, limitations: list[str],
    plan: VisualizationPlan, language: str,
) -> dict[str, Any]:
    return {
        "selected_facts": [{
            "fact_type": item.fact_type, "values": item.values,
            "caveats": item.caveats,
        } for item in facts],
        "why_this_chart": rationale,
        "limitations": limitations,
        "semantic_labels": {
            "x": plan.x_encoding.semantic_label if plan.x_encoding else "",
            "y": plan.y_encoding.semantic_label if plan.y_encoding else "",
        },
        "requested_language": language,
    }


def _canonical_number(value: str) -> str:
    suffix = "%" if value.endswith("%") else ""
    raw = value.removesuffix("%").replace(",", "").lstrip("+")
    try:
        raw = format(float(raw), ".12g")
    except ValueError:
        pass
    return raw + suffix


def validate_draft(draft: XAIModelDraft, context: dict[str, Any]) -> bool:
    if draft.language.casefold() != str(context["requested_language"]).casefold():
        return False
    prose = f"{draft.why_this_chart} {draft.what_it_shows}"
    if _CAUSAL.search(prose) or _SIGNIFICANCE.search(prose) or _BUSINESS.search(prose) or _INTERNAL.search(prose):
        return False
    if any(item not in context["limitations"] for item in draft.limitations):
        return False
    facts = context["selected_facts"]
    types = {item["fact_type"] for item in facts}
    if re.search(r"\bcorrelat", prose, re.I) and "CORRELATION" not in types:
        return False
    if re.search(r"\b(?:trend|increased|decreased|change)\b", draft.what_it_shows, re.I) and not types.intersection({"TREND_DIRECTION", "ABSOLUTE_CHANGE", "PERCENT_CHANGE", "LARGEST_INCREASE", "LARGEST_DECREASE"}):
        return False
    if re.search(r"\boutliers?\b", draft.what_it_shows, re.I) and "OUTLIER_COUNT" not in types:
        return False
    serialized = json.dumps(context, ensure_ascii=False, sort_keys=True)
    allowed_numbers = {_canonical_number(match.group(0)) for match in _NUMBER.finditer(serialized)}
    if any(_canonical_number(match.group(0)) not in allowed_numbers for match in _NUMBER.finditer(" ".join([prose, *draft.limitations]))):
        return False
    allowed_words = set(_CAPITAL.findall(serialized)) | _SAFE
    if any(token not in allowed_words for token in _CAPITAL.findall(prose)):
        return False
    return True


def generate_wording(
    facts: list[XAIFact], plan: VisualizationPlan, limitations: list[str],
    provider: LLMProvider | None, *, language: str,
) -> tuple[str, str, list[str], str, bool]:
    rationale = chart_rationale(plan.chart_family)
    fallback = deterministic_findings(facts)
    if provider is None:
        return rationale, fallback, limitations, "en", True
    context = serialize_for_provider(facts, rationale, limitations, plan, language)
    prompt = f"{XAI_SYSTEM_PROMPT}\n\nVerified XAI context:\n{json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
    try:
        draft = XAIModelDraft.model_validate(provider.generate_structured(prompt, XAIModelDraft))
        if not validate_draft(draft, context):
            raise ValueError("unverified XAI wording")
        return draft.why_this_chart, draft.what_it_shows, draft.limitations, draft.language, False
    except Exception:
        return rationale, fallback, limitations, "en", True


__all__ = [
    "XAI_SYSTEM_PROMPT", "chart_rationale", "deterministic_findings",
    "generate_wording", "selection_rationale", "serialize_for_provider",
    "validate_draft",
]
