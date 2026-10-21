"""Metadata-only analytical opportunity discovery and typed plan construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pipelines.pipeline3_results.grounding.engine import (
    is_category_eligible,
    is_identifier_eligible,
    is_numeric_eligible,
    is_temporal_eligible,
)
from pipelines.pipeline3_results.grounding.schemas import (
    AnalyticalExpression,
    DatasetColumnKnowledge,
    DatasetKnowledge,
)
from pipelines.pipeline3_results.planning.schemas import (
    AnalyticalPlan,
    GroupingExpression,
    PlanProvenance,
    SortExpression,
)


@dataclass(frozen=True)
class AnalyticalOpportunities:
    measures: tuple[DatasetColumnKnowledge, ...]
    dimensions: tuple[DatasetColumnKnowledge, ...]
    identifiers: tuple[DatasetColumnKnowledge, ...]
    temporal: tuple[DatasetColumnKnowledge, ...]
    primary_measure: DatasetColumnKnowledge | None
    primary_dimension: DatasetColumnKnowledge | None


def _coverage(column: DatasetColumnKnowledge) -> float:
    rows = column.quality_metadata.get("row_count")
    non_null = column.quality_metadata.get("non_null_count")
    if isinstance(rows, (int, float)) and rows > 0 and isinstance(non_null, (int, float)):
        return max(0.0, min(1.0, float(non_null) / float(rows)))
    return 0.75


def _score(column: DatasetColumnKnowledge) -> float:
    confidence = column.confidence if column.confidence is not None else 0.5
    evidence = 0.1 if column.concept_description else 0.0
    return (confidence * 0.6) + (_coverage(column) * 0.3) + evidence


def _ordered(columns: list[DatasetColumnKnowledge]) -> tuple[DatasetColumnKnowledge, ...]:
    return tuple(sorted(columns, key=lambda item: (-_score(item), item.name.casefold())))


def _unambiguous(columns: tuple[DatasetColumnKnowledge, ...]) -> DatasetColumnKnowledge | None:
    if not columns:
        return None
    if len(columns) == 1:
        return columns[0]
    if _score(columns[0]) - _score(columns[1]) < 0.05:
        return None
    return columns[0]


def discover_opportunities(knowledge: DatasetKnowledge) -> AnalyticalOpportunities:
    identifiers = [item for item in knowledge.columns if is_identifier_eligible(item)]
    temporal = [item for item in knowledge.columns if is_temporal_eligible(item)]
    measures = [
        item for item in knowledge.columns
        if is_numeric_eligible(item)
        and not is_identifier_eligible(item)
        and not is_temporal_eligible(item)
    ]
    dimensions = [
        item for item in knowledge.columns
        if is_category_eligible(item)
        and not is_identifier_eligible(item)
        and not is_temporal_eligible(item)
    ]
    ordered_measures = _ordered(measures)
    ordered_dimensions = _ordered(dimensions)
    return AnalyticalOpportunities(
        measures=ordered_measures,
        dimensions=ordered_dimensions,
        identifiers=_ordered(identifiers),
        temporal=_ordered(temporal),
        primary_measure=_unambiguous(ordered_measures),
        primary_dimension=_unambiguous(ordered_dimensions),
    )


def display_label(column: DatasetColumnKnowledge) -> str:
    description = (column.concept_description or "").strip()
    if description and len(description) <= 80 and "\n" not in description:
        return description
    return column.name.replace("_", " ")


def make_plan(
    operation: str,
    *,
    measures: list[AnalyticalExpression],
    dimension: str | None = None,
    temporal: str | None = None,
    comparison: str = "NONE",
    limit: int | None = None,
) -> AnalyticalPlan:
    payload = {
        "operation": operation,
        "measures": [item.model_dump(mode="json") for item in measures],
        "dimension": dimension,
        "temporal": temporal,
        "comparison": comparison,
        "limit": limit,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    sort = []
    if operation in {"RANK", "TOP_N", "BOTTOM_N"}:
        sort = [SortExpression(
            target="MEASURE",
            direction="ASC" if comparison in {"LOWEST", "BOTTOM_N"} else "DESC",
            measure_index=0,
        )]
    return AnalyticalPlan(
        plan_id=f"plan_{digest}",
        operation=operation,
        measures=measures,
        dimensions=(
            [GroupingExpression(kind="GROUP_BY", column=dimension)]
            if dimension else []
        ),
        temporal=(
            [GroupingExpression(kind="TIME_BUCKET", column=temporal, time_granularity="MONTH")]
            if temporal else []
        ),
        sort=sort,
        limit=limit,
        comparison=comparison,
        provenance=PlanProvenance(
            grounding_status="RESOLVED",
            planner_decisions=["phase6:metadata_derived_opportunity"],
        ),
    )


__all__ = [
    "AnalyticalOpportunities", "discover_opportunities", "display_label", "make_plan",
]
