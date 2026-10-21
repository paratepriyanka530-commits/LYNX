"""Deterministic dataset-summary facts backed by the shared executor."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import pandas as pd
from pydantic import JsonValue

from pipelines.pipeline3_results.execution.pandas_executor import PandasAnalyticalExecutor
from pipelines.pipeline3_results.execution.schemas import VerifiedResult
from pipelines.pipeline3_results.grounding.schemas import (
    AnalyticalExpression,
    DatasetKnowledge,
)
from pipelines.pipeline3_results.insights.landing.opportunities import (
    AnalyticalOpportunities,
    discover_opportunities,
    display_label,
    make_plan,
)
from pipelines.pipeline3_results.insights.landing.schemas import (
    SummaryFact,
    SummaryPackage,
)
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan
from pipelines.pipeline3_results.planning.validator import validate_analytical_plan


def _fact(
    fact_type: str,
    importance: float,
    values: dict[str, JsonValue],
    *,
    evidence_refs: list[str] | None = None,
    caveats: list[str] | None = None,
) -> SummaryFact:
    identity = json.dumps(
        {"type": fact_type, "values": values},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return SummaryFact(
        fact_id=f"fact_{digest}",
        fact_type=fact_type,  # type: ignore[arg-type]
        importance=importance,
        values=values,
        evidence_refs=evidence_refs or [],
        caveats=caveats or [],
    )


def _execute(
    plan: AnalyticalPlan,
    dataframe: pd.DataFrame,
    knowledge: DatasetKnowledge,
) -> VerifiedResult | None:
    validation = validate_analytical_plan(plan, knowledge)
    if validation.status != "READY":
        return None
    result = PandasAnalyticalExecutor().execute(plan, dataframe, knowledge)
    if result.status != "READY" or result.verification.status not in {
        "VERIFIED", "VERIFIED_WITH_WARNINGS",
    }:
        return None
    return result


def _quality_fact(knowledge: DatasetKnowledge) -> SummaryFact | None:
    if not knowledge.metadata_availability.get("pipeline1_report"):
        return None
    values: dict[str, JsonValue] = {}
    score = knowledge.quality.get("quality_score")
    if isinstance(score, (int, float)):
        values["quality_score"] = float(score)
    issues = knowledge.quality.get("total_issues_detected")
    if isinstance(issues, int):
        values["reported_issue_count"] = issues

    missing_columns = 0
    missing_cells = 0
    observed_cells = 0
    for column in knowledge.columns:
        missing = column.quality_metadata.get("missing")
        if isinstance(missing, dict):
            missing = missing.get("count")
        rows = column.quality_metadata.get("row_count")
        if isinstance(rows, int) and rows >= 0:
            observed_cells += rows
        if isinstance(missing, (int, float)) and missing > 0:
            missing_columns += 1
            missing_cells += int(missing)
    if missing_columns:
        values["columns_with_missing_values"] = missing_columns
        values["reported_missing_cells"] = missing_cells
        if observed_cells:
            values["reported_missing_percentage"] = (
                float(missing_cells) / float(observed_cells) * 100.0
            )
    if not values:
        return None
    return _fact(
        "DATA_QUALITY", 0.88, values,
        evidence_refs=["dataset_knowledge:quality_metadata"],
    )


def _semantic_fact(opportunities: AnalyticalOpportunities) -> SummaryFact | None:
    values: dict[str, JsonValue] = {
        "measures": [display_label(item) for item in opportunities.measures[:3]],
        "dimensions": [display_label(item) for item in opportunities.dimensions[:3]],
        "temporal_fields": [display_label(item) for item in opportunities.temporal[:2]],
        "entity_identifiers": [display_label(item) for item in opportunities.identifiers[:2]],
    }
    if not any(values.values()):
        return None
    return _fact(
        "SEMANTIC_STRUCTURE", 0.76, values,
        evidence_refs=["dataset_knowledge:semantic_fields"],
    )


def _computed_facts(
    dataframe: pd.DataFrame,
    knowledge: DatasetKnowledge,
    opportunities: AnalyticalOpportunities,
) -> list[SummaryFact]:
    facts: list[SummaryFact] = []
    measure = opportunities.primary_measure
    if measure is not None:
        average_plan = make_plan(
            "AGGREGATE",
            measures=[AnalyticalExpression(
                kind="AGGREGATION", operation="MEAN", column=measure.name,
            )],
        )
        average = _execute(average_plan, dataframe, knowledge)
        if average is not None:
            facts.append(_fact(
                "MEASURE_AGGREGATE", 0.93,
                {
                    "column": measure.name,
                    "label": display_label(measure),
                    "aggregation": "MEAN",
                    "value": average.scalar,
                    "usable_rows": average.coverage.usable_rows,
                },
                evidence_refs=[average.plan_id, average.result_id],
                caveats=list(average.warnings),
            ))
        range_plan = make_plan(
            "STATISTIC",
            measures=[AnalyticalExpression(
                kind="STATISTIC", operation="RANGE", column=measure.name,
            )],
        )
        spread = _execute(range_plan, dataframe, knowledge)
        if spread is not None:
            facts.append(_fact(
                "NUMERIC_RANGE", 0.72,
                {
                    "column": measure.name,
                    "label": display_label(measure),
                    "statistic": "RANGE",
                    "value": spread.scalar,
                    "definition": "maximum minus minimum",
                },
                evidence_refs=[spread.plan_id, spread.result_id],
            ))

    dimension = opportunities.primary_dimension
    if dimension is not None:
        rank_plan = make_plan(
            "RANK",
            measures=[AnalyticalExpression(kind="COUNT_ROWS")],
            dimension=dimension.name,
            comparison="HIGHEST",
        )
        ranked = _execute(rank_plan, dataframe, knowledge)
        if ranked is not None and ranked.rows:
            facts.append(_fact(
                "GROUP_RANKING", 0.95,
                {
                    "dimension": dimension.name,
                    "label": display_label(dimension),
                    "comparison": "HIGHEST",
                    "count_semantics": "COUNT_ROWS",
                    "leaders": [row[dimension.name] for row in ranked.rows],
                    "value": ranked.rows[0]["value"],
                    "tie": len(ranked.rows) > 1,
                },
                evidence_refs=[ranked.plan_id, ranked.result_id],
            ))

    if len(opportunities.identifiers) == 1:
        identifier = opportunities.identifiers[0]
        count_plan = make_plan(
            "COUNT",
            measures=[AnalyticalExpression(
                kind="COUNT_DISTINCT", operation="COUNT_DISTINCT",
                column=identifier.name,
            )],
        )
        count = _execute(count_plan, dataframe, knowledge)
        if count is not None:
            facts.append(_fact(
                "ENTITY_COUNT", 0.87,
                {
                    "column": identifier.name,
                    "label": display_label(identifier),
                    "count_semantics": "COUNT_DISTINCT",
                    "value": count.scalar,
                },
                evidence_refs=[count.plan_id, count.result_id],
            ))
    return facts


def _opportunity_facts(
    opportunities: AnalyticalOpportunities,
) -> tuple[list[SummaryFact], list[str]]:
    facts: list[SummaryFact] = []
    names: list[str] = []
    if opportunities.temporal and opportunities.measures:
        temporal = opportunities.temporal[0]
        facts.append(_fact(
            "TREND_OPPORTUNITY", 0.83,
            {"temporal_field": temporal.name, "label": display_label(temporal)},
            evidence_refs=["dataset_knowledge:temporal_measure_compatibility"],
        ))
        names.append("TREND")
    if len(opportunities.measures) >= 2:
        first, second = opportunities.measures[:2]
        facts.append(_fact(
            "RELATIONSHIP_OPPORTUNITY", 0.81,
            {
                "first_measure": first.name,
                "first_label": display_label(first),
                "second_measure": second.name,
                "second_label": display_label(second),
            },
            evidence_refs=["dataset_knowledge:numeric_pair_compatibility"],
        ))
        names.append("RELATIONSHIP")
    if opportunities.dimensions:
        dimension = opportunities.dimensions[0]
        facts.append(_fact(
            "CATEGORY_OPPORTUNITY", 0.79,
            {"dimension": dimension.name, "label": display_label(dimension)},
            evidence_refs=["dataset_knowledge:grouping_compatibility"],
        ))
        names.append("CATEGORY_COMPARISON")
    if opportunities.identifiers:
        names.append("DISTINCT_ENTITY_COUNT")
    return facts, names


_DIVERSITY_FAMILY = {
    "DATASET_OVERVIEW": "OVERVIEW",
    "DATA_QUALITY": "QUALITY",
    "SEMANTIC_STRUCTURE": "STRUCTURE",
    "MEASURE_AGGREGATE": "FINDING",
    "GROUP_RANKING": "FINDING",
    "ENTITY_COUNT": "ENTITY",
    "NUMERIC_RANGE": "FINDING",
    "TREND_OPPORTUNITY": "OPPORTUNITY",
    "RELATIONSHIP_OPPORTUNITY": "OPPORTUNITY",
    "CATEGORY_OPPORTUNITY": "OPPORTUNITY",
}


def select_summary_facts(
    candidates: Iterable[SummaryFact], *, maximum: int = 5
) -> list[SummaryFact]:
    selected: list[SummaryFact] = []
    counts: dict[str, int] = {}
    limits = {"FINDING": 1, "OPPORTUNITY": 1}
    for fact in sorted(candidates, key=lambda item: (-item.importance, item.fact_id)):
        family = _DIVERSITY_FAMILY[fact.fact_type]
        if counts.get(family, 0) >= limits.get(family, 1):
            continue
        selected.append(fact)
        counts[family] = counts.get(family, 0) + 1
        if len(selected) == maximum:
            break
    return selected


def build_summary_package(
    dataframe: pd.DataFrame,
    knowledge: DatasetKnowledge,
    *,
    dataset_id: str | None = None,
) -> SummaryPackage:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    opportunities = discover_opportunities(knowledge)
    candidates = [_fact(
        "DATASET_OVERVIEW", 1.0,
        {"row_count": len(dataframe), "column_count": len(dataframe.columns)},
        evidence_refs=["cleaned_dataset:shape"],
    )]
    quality = _quality_fact(knowledge)
    if quality is not None:
        candidates.append(quality)
    semantic = _semantic_fact(opportunities)
    if semantic is not None:
        candidates.append(semantic)
    candidates.extend(_computed_facts(dataframe, knowledge, opportunities))
    opportunity_facts, opportunity_names = _opportunity_facts(opportunities)
    candidates.extend(opportunity_facts)
    facts = select_summary_facts(candidates)
    return SummaryPackage(
        dataset_id=dataset_id,
        status="READY" if facts else "EMPTY",
        facts=facts,
        opportunities=opportunity_names,
        warnings=[],
    )


__all__ = ["build_summary_package", "select_summary_facts"]
