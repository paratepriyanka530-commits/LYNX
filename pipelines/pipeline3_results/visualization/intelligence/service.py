"""Automatic and requested visualization intelligence over the shared executor."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import pandas as pd

from pipelines.pipeline3_results.execution.pandas_executor import PandasAnalyticalExecutor
from pipelines.pipeline3_results.execution.schemas import VerifiedResult
from pipelines.pipeline3_results.grounding.dataset_knowledge import build_dataset_knowledge
from pipelines.pipeline3_results.grounding.schemas import AnalyticalExpression, DatasetKnowledge
from pipelines.pipeline3_results.insights.landing.opportunities import discover_opportunities, display_label, make_plan
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan
from pipelines.pipeline3_results.planning.validator import validate_analytical_plan
from .planner import build_visualization
from .schemas import ChartFamily, DashboardVisualization, KPICandidate, ScoreComponent, VisualizationDashboard
from .suitability import assess_suitability, default_family, purpose_for


def _execute(plan: AnalyticalPlan, dataframe: pd.DataFrame, knowledge: DatasetKnowledge) -> VerifiedResult | None:
    if validate_analytical_plan(plan, knowledge).status != "READY":
        return None
    result = PandasAnalyticalExecutor().execute(plan, dataframe, knowledge)
    if result.status != "READY" or result.verification.status not in {"VERIFIED", "VERIFIED_WITH_WARNINGS"}:
        return None
    return result


def _confidence(plan: AnalyticalPlan, knowledge: DatasetKnowledge) -> float:
    names = [m.column for m in plan.measures if m.column] + [d.column for d in plan.dimensions] + [t.column for t in plan.temporal]
    values = [(knowledge.get_column(name).confidence or 0.5) for name in names if knowledge.get_column(name)]
    return sum(values) / len(values) if values else 0.75


def _score(plan: AnalyticalPlan, result: VerifiedResult, knowledge: DatasetKnowledge, family: ChartFamily) -> tuple[float, list[ScoreComponent]]:
    confidence = _confidence(plan, knowledge)
    coverage = result.coverage.usable_rows / result.coverage.source_rows if result.coverage.source_rows else 0.0
    suitability = assess_suitability(plan, result, knowledge, family)
    suitable = 1.0 if suitability.status == "SUITABLE" else 0.75
    cardinality = 1.0
    if family in {ChartFamily.PIE, ChartFamily.DOUGHNUT}:
        cardinality = 1.0 if len(result.rows) <= 6 else 0.75
    usefulness = {"RANK": .95, "TREND": .95, "RELATIONSHIP": .9, "PART_TO_WHOLE": .82, "DISTRIBUTION": .85}.get(plan.operation.value, .8)
    components = [
        ScoreComponent(name="semantic_importance", value=round(confidence, 4), reason_code="METADATA_CONFIDENCE"),
        ScoreComponent(name="data_coverage", value=round(coverage, 4), reason_code="VERIFIED_USABLE_ROW_COVERAGE"),
        ScoreComponent(name="chart_suitability", value=suitable, reason_code=suitability.reason_codes[0]),
        ScoreComponent(name="cardinality", value=cardinality, reason_code="FAMILY_CARDINALITY_FIT"),
        ScoreComponent(name="analytical_usefulness", value=usefulness, reason_code=f"{plan.operation.value}_USEFULNESS"),
    ]
    score = confidence * .25 + coverage * .25 + suitable * .2 + cardinality * .1 + usefulness * .2
    return round(score, 6), components


def _candidate_plans(knowledge: DatasetKnowledge) -> list[AnalyticalPlan]:
    o = discover_opportunities(knowledge)
    measures = [item for item in o.measures if (item.confidence or .5) >= .5]
    dimensions = [item for item in o.dimensions if (item.confidence or .5) >= .5]
    plans: list[AnalyticalPlan] = []
    if measures and o.temporal:
        plans.append(make_plan("TREND", measures=[AnalyticalExpression(kind="AGGREGATION", operation="MEAN", column=measures[0].name)], temporal=o.temporal[0].name))
    if measures and dimensions:
        plans.append(make_plan("GROUP_AGGREGATE", measures=[AnalyticalExpression(kind="AGGREGATION", operation="MEAN", column=measures[0].name)], dimension=dimensions[0].name))
    if dimensions:
        plans.append(make_plan("RANK", measures=[AnalyticalExpression(kind="COUNT_ROWS")], dimension=dimensions[0].name, comparison="HIGHEST"))
        plans.append(make_plan("PART_TO_WHOLE", measures=[AnalyticalExpression(kind="COUNT_ROWS")], dimension=dimensions[0].name))
    if measures:
        plans.append(make_plan("DISTRIBUTION", measures=[AnalyticalExpression(kind="COLUMN", column=measures[0].name)]))
    if len(measures) >= 2:
        plans.append(make_plan("RELATIONSHIP", measures=[AnalyticalExpression(kind="COLUMN", column=measures[0].name), AnalyticalExpression(kind="COLUMN", column=measures[1].name)]))
    return plans


def _kpis(dataframe: pd.DataFrame, knowledge: DatasetKnowledge) -> list[KPICandidate]:
    o = discover_opportunities(knowledge)
    definitions: list[tuple[str, str, AnalyticalPlan, float, str]] = [(
        "Record Count", "INTEGER", make_plan("COUNT", measures=[AnalyticalExpression(kind="COUNT_ROWS")]), 1.0, "COUNT_ROWS",
    )]
    if o.primary_measure is not None:
        definitions.append((f"Average {display_label(o.primary_measure)}", "NUMBER", make_plan("AGGREGATE", measures=[AnalyticalExpression(kind="AGGREGATION", operation="MEAN", column=o.primary_measure.name)]), .9, "MEAN"))
    if len(o.identifiers) == 1:
        identifier = o.identifiers[0]
        definitions.append((f"Distinct {display_label(identifier)}", "INTEGER", make_plan("COUNT", measures=[AnalyticalExpression(kind="COUNT_DISTINCT", operation="COUNT_DISTINCT", column=identifier.name)]), .86, "COUNT_DISTINCT"))
    if o.primary_dimension is not None:
        dimension = o.primary_dimension
        definitions.append((f"Largest {display_label(dimension)} Group", "INTEGER", make_plan("RANK", measures=[AnalyticalExpression(kind="COUNT_ROWS")], dimension=dimension.name, comparison="HIGHEST"), .82, "RANK_COUNT"))
    items: list[KPICandidate] = []
    for label, fmt, plan, importance, kind in definitions[:4]:
        result = _execute(plan, dataframe, knowledge)
        if result is None:
            continue
        value = result.scalar if result.scalar is not None else (result.rows[0].get("value") if result.rows else None)
        if value is None:
            continue
        digest = hashlib.sha256(f"{plan.plan_id}:{kind}".encode()).hexdigest()[:16]
        items.append(KPICandidate(id=f"kpi_{digest}", label=label, expression={"kind": kind, "plan_id": plan.plan_id}, verified_result_id=result.result_id, value=value, format_hint=fmt, importance=importance, evidence=[plan.plan_id, result.result_id, "shared_executor:verified"]))
    return items


def build_requested_visualization(plan: AnalyticalPlan, result: VerifiedResult, knowledge: DatasetKnowledge, requested_family: ChartFamily | None = None) -> tuple[Any, DashboardVisualization | None]:
    family = requested_family or default_family(purpose_for(plan, result))
    assessment = assess_suitability(plan, result, knowledge, family)
    return assessment, build_visualization(plan, result, knowledge, source="USER_REQUESTED", requested_family=family)


def plan_from_validated_request(request) -> AnalyticalPlan:
    """Compatibility conversion only; validation/NLP remain authoritative upstream."""
    chart = request.chart_type
    if chart == "SCATTER":
        return make_plan("RELATIONSHIP", measures=[
            AnalyticalExpression(kind="COLUMN", column=request.x_column),
            AnalyticalExpression(kind="COLUMN", column=request.y_column),
        ])
    if chart == "HISTOGRAM":
        return make_plan("DISTRIBUTION", measures=[AnalyticalExpression(kind="COLUMN", column=request.x_column)])
    if chart == "COUNT_BAR":
        return make_plan("DISTRIBUTION", measures=[AnalyticalExpression(kind="COLUMN", column=request.x_column)])
    if chart == "LINE":
        return make_plan("TREND", measures=[AnalyticalExpression(
            kind="AGGREGATION", operation=request.aggregation or "MEAN", column=request.y_column,
        )], temporal=request.x_column)
    operation = "PART_TO_WHOLE" if chart == "PIE" else "GROUP_AGGREGATE"
    expression = (
        AnalyticalExpression(kind="COUNT_ROWS")
        if request.y_column is None
        else AnalyticalExpression(kind="AGGREGATION", operation=request.aggregation or "SUM", column=request.y_column)
    )
    return make_plan(operation, measures=[expression], dimension=request.x_column)


class VisualizationIntelligenceService:
    def __init__(self) -> None:
        # Request-scoped evidence sidecar for downstream XAI. This is not part
        # of the dashboard contract and does not alter recommendation scoring.
        self._verified_results: dict[str, VerifiedResult] = {}

    def get_verified_result(self, result_id: str) -> VerifiedResult | None:
        return self._verified_results.get(result_id)

    def build_dashboard(self, dataset_id: str, dataframe: pd.DataFrame, *, pipeline1_report: Mapping[str, Any] | None = None, max_visualizations: int = 5) -> VisualizationDashboard:
        self._verified_results.clear()
        knowledge = build_dataset_knowledge(pipeline1_report, dataset_id=dataset_id, dataframe_schema={str(name): str(dtype) for name, dtype in dataframe.dtypes.items()})
        candidates: list[DashboardVisualization] = []
        seen: set[tuple[str, str]] = set()
        for plan in _candidate_plans(knowledge):
            result = _execute(plan, dataframe, knowledge)
            if result is None:
                continue
            family = default_family(purpose_for(plan, result))
            score, components = _score(plan, result, knowledge, family)
            visual = build_visualization(plan, result, knowledge, source="AUTO", requested_family=family, score=score, score_components=components)
            if visual is None:
                continue
            self._verified_results[result.result_id] = result
            key = (visual.plan.purpose.value, plan.plan_id)
            if key not in seen:
                candidates.append(visual)
                seen.add(key)
        selected: list[DashboardVisualization] = []
        purposes: set[str] = set()
        for visual in sorted(candidates, key=lambda item: (-(item.score or 0), item.id)):
            if visual.plan.purpose.value in purposes:
                continue
            selected.append(visual)
            purposes.add(visual.plan.purpose.value)
            if len(selected) == min(5, max_visualizations):
                break
        kpis = _kpis(dataframe, knowledge)
        return VisualizationDashboard(dataset_id=dataset_id, status="READY" if selected or kpis else "EMPTY", kpis=kpis, visualizations=selected, warnings=[])


__all__ = ["VisualizationIntelligenceService", "build_requested_visualization", "plan_from_validated_request"]
