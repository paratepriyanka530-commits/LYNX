"""Build bounded, display-ready response facts from VerifiedResult."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

from pydantic import JsonValue

from pipelines.pipeline3_results.execution.schemas import VerifiedResult
from pipelines.pipeline3_results.grounding.schemas import DatasetKnowledge
from pipelines.pipeline3_results.insights.response_generation.schemas import (
    ResponseContext,
)
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan


MAX_RESPONSE_ROWS = 25


def format_number(value: int | float) -> str:
    """Format without changing the represented numeric value."""
    if isinstance(value, bool):
        raise TypeError("boolean values are not numeric response facts")
    decimal = Decimal(str(value))
    rendered = format(decimal, "f")
    if isinstance(value, float):
        if "." not in rendered:
            rendered = f"{rendered}.00"
        else:
            integer_part, fraction_part = rendered.split(".", 1)
            rendered = f"{integer_part}.{fraction_part.ljust(2, '0')}"
    integer, dot, fraction = rendered.partition(".")
    sign = ""
    if integer.startswith("-"):
        sign, integer = "-", integer[1:]
    grouped = f"{int(integer or '0'):,}"
    return f"{sign}{grouped}{dot}{fraction}" if dot else f"{sign}{grouped}"


def format_percentage(value: int | float) -> str:
    rendered = format(Decimal(str(value)) * Decimal("100"), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return f"{rendered}%"


def _display_value(name: str, value: JsonValue) -> JsonValue:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format_number(value)
    return value


def _display_structure(value: JsonValue) -> JsonValue:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format_number(value)
    if isinstance(value, list):
        return [_display_structure(item) for item in value]
    if isinstance(value, dict):
        return {name: _display_structure(item) for name, item in value.items()}
    return value


def _labels(plan: AnalyticalPlan, knowledge: DatasetKnowledge) -> dict[str, str]:
    columns = {
        expression.column for expression in plan.measures if expression.column
    }
    columns.update(item.column for item in [*plan.dimensions, *plan.temporal])
    labels: dict[str, str] = {}
    for name in sorted(columns):
        column = knowledge.get_column(name)
        if column is None:
            continue
        description = (column.concept_description or "").strip()
        if description and len(description) <= 80 and "\n" not in description:
            labels[name] = description
        else:
            labels[name] = name.replace("_", " ")
    return labels


def _units(plan: AnalyticalPlan, knowledge: DatasetKnowledge) -> dict[str, str]:
    units: dict[str, str] = {}
    for expression in plan.measures:
        if expression.column is None:
            continue
        column = knowledge.get_column(expression.column)
        if column is None:
            continue
        for key in ("currency", "unit", "format"):
            value = column.quality_metadata.get(key)
            if isinstance(value, str) and value.strip():
                units[expression.column] = value.strip()
                break
    return units


def serialize_verified_result(
    plan: AnalyticalPlan,
    result: VerifiedResult,
    *,
    max_rows: int = MAX_RESPONSE_ROWS,
) -> tuple[dict[str, JsonValue], bool]:
    """Serialize only verified, bounded facts; never serialize source records."""
    rows = result.rows[:max_rows]
    serialized_rows: list[dict[str, JsonValue]] = []
    for row in rows:
        rendered: dict[str, JsonValue] = {}
        for name, value in row.items():
            rendered[name] = _display_value(name, value)
            if name == "share_of_total" and isinstance(value, (int, float)):
                rendered["share_percent_display"] = format_percentage(value)
        serialized_rows.append(rendered)

    expression = plan.measures[0] if plan.measures else None
    metric = expression.column if expression and expression.column else expression.kind if expression else None
    payload: dict[str, JsonValue] = {
        "status": result.status,
        "result_type": result.result_type,
        "operation": str(plan.operation),
        "metric": metric,
        "aggregation_or_statistic": expression.operation if expression else None,
        "count_mode": expression.kind if expression and expression.kind.startswith("COUNT") else None,
        "comparison": plan.comparison,
        "scalar_display": (
            _display_value("value", result.scalar)
            if result.scalar is not None else None
        ),
        "rows": serialized_rows,
        "coverage": result.coverage.model_dump(mode="json"),
        "facts": _display_structure(result.execution_metadata.details),
        "row_count": len(result.rows),
        "rows_included": len(serialized_rows),
        "rows_truncated": max(0, len(result.rows) - len(serialized_rows)),
    }
    return payload, len(result.rows) > len(serialized_rows)


def build_response_context(
    original_query: str,
    plan: AnalyticalPlan,
    result: VerifiedResult,
    knowledge: DatasetKnowledge,
    *,
    requested_language: str = "en",
    conversation_context: Mapping[str, Any] | None = None,
) -> ResponseContext:
    serialized, truncated = serialize_verified_result(plan, result)
    expression = plan.measures[0] if plan.measures else None
    semantic_summary: dict[str, JsonValue] = {
        "operation": str(plan.operation),
        "measure": expression.column if expression else None,
        "aggregation": expression.operation if expression else None,
        "count_mode": expression.kind if expression and expression.kind.startswith("COUNT") else None,
        "dimensions": [item.column for item in plan.dimensions],
        "temporal": [
            {"column": item.column, "granularity": item.time_granularity}
            for item in plan.temporal
        ],
        "comparison": plan.comparison,
        "limit": plan.limit,
    }
    bounded_conversation: dict[str, JsonValue] = {}
    for key in ("conversation_id", "previous_verified_result_id", "response_language"):
        value = (conversation_context or {}).get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            bounded_conversation[key] = value
    warnings = list(result.warnings)
    if truncated:
        warnings.append(
            f"Response context includes {MAX_RESPONSE_ROWS} of {len(result.rows)} verified rows."
        )
    return ResponseContext(
        original_query=original_query.strip(),
        semantic_intent_summary=semantic_summary,
        verified_result=serialized,
        semantic_labels=_labels(plan, knowledge),
        units_or_formats_if_known=_units(plan, knowledge),
        warnings=warnings,
        conversation_context=bounded_conversation,
        requested_language=requested_language,
        truncated=truncated,
    )


__all__ = [
    "MAX_RESPONSE_ROWS",
    "build_response_context",
    "format_number",
    "format_percentage",
    "serialize_verified_result",
]
