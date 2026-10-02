"""Adapt model-produced concepts to the existing deterministic grounding path."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from pipelines.pipeline3_results.nlp.ambiguity_detector import detect_ambiguities
from pipelines.pipeline3_results.nlp.column_resolver import resolve_columns
from pipelines.pipeline3_results.nlp.experimental.schemas import (
    AnalyticalInterpretation,
    GroundedConcept,
    GroundedInterpretation,
)
from pipelines.pipeline3_results.nlp.schemas import ExtractedEntities, QueryFilter
from pipelines.pipeline3_results.visualization.column_classifier import (
    classify_dataframe,
)
from pipelines.pipeline3_results.visualization.schemas import DatasetClassification


def _production_chart(
    requested_chart: str | None,
) -> tuple[str | None, str | None]:
    if requested_chart == "DOUGHNUT":
        return "PIE", "DOUGHNUT"
    return requested_chart, None


def ground_interpretation(
    interpretation: AnalyticalInterpretation,
    dataframe: pd.DataFrame,
    *,
    pipeline1_report: Mapping[str, Any] | None = None,
    classification: DatasetClassification | None = None,
) -> GroundedInterpretation:
    """Ground concepts without changing the production resolver or inspecting rows.

    The existing classifier may inspect values to assign analytical roles, as it
    already does in production. Raw values are never supplied to the language
    model and this function performs no analytical computation.
    """
    if not isinstance(interpretation, AnalyticalInterpretation):
        raise TypeError("interpretation must be AnalyticalInterpretation")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if interpretation.unsupported:
        return GroundedInterpretation(
            status="UNSUPPORTED",
            reason=interpretation.unsupported_reason or "unsupported interpretation",
        )

    active_classification = classification or classify_dataframe(
        dataframe,
        pipeline1_report=pipeline1_report,
    )
    chart, variant = _production_chart(interpretation.requested_chart)
    filters = [
        QueryFilter(
            raw_text=f"{item.field_concept} {item.operator} {item.value_concept or ''}".strip(),
            column_term=item.field_concept,
            operator=(
                item.operator
                if item.operator in {
                    "EQUALS", "NOT_EQUALS", "GREATER_THAN",
                    "GREATER_THAN_OR_EQUAL", "LESS_THAN", "LESS_THAN_OR_EQUAL",
                    "CONTAINS", "UNKNOWN",
                }
                else "UNKNOWN"
            ),
            value_term=item.value_concept,
        )
        for item in interpretation.filters
    ]
    time_terms = (
        [interpretation.time_expression.expression]
        if interpretation.time_expression is not None
        else []
    )
    entities = ExtractedEntities(
        measure_terms=list(interpretation.measure_concepts),
        dimension_terms=list(interpretation.dimension_concepts),
        aggregation=interpretation.aggregation,
        comparison=interpretation.comparison,
        requested_chart=chart,
        requested_chart_variant=variant,
        filters=filters,
        time_terms=time_terms,
        limit=interpretation.limit,
        signals=("experimental_pretrained_interpretation",),
    )
    resolutions = resolve_columns(
        entities,
        dataframe,
        pipeline1_report=pipeline1_report,
        classification=active_classification,
    )
    decisions = detect_ambiguities(resolutions)
    concepts: list[GroundedConcept] = []
    measures: list[str] = []
    dimensions: list[str] = []
    times: list[str] = []
    for resolution, decision in zip(resolutions, decisions.decisions, strict=True):
        evidence = {
            candidate.column_name: list(candidate.evidence_sources)
            for candidate in resolution.candidates
        }
        concepts.append(GroundedConcept(
            concept=resolution.term,
            kind=("MEASURE" if resolution.kind == "RELATIONSHIP" else resolution.kind),
            status=decision.status,
            resolved_column=decision.resolved_column,
            candidate_columns=[candidate.column_name for candidate in resolution.candidates],
            reason=decision.reason,
            evidence=evidence,
        ))
        if decision.status != "RESOLVED" or decision.resolved_column is None:
            continue
        if resolution.kind in {"MEASURE", "RELATIONSHIP"}:
            measures.append(decision.resolved_column)
        elif resolution.kind == "DIMENSION":
            dimensions.append(decision.resolved_column)
        elif resolution.kind == "TIME":
            times.append(decision.resolved_column)

    if interpretation.requires_clarification:
        status = "NEEDS_CLARIFICATION"
        reason = interpretation.clarification_reason or "semantic clarification required"
    elif not resolutions:
        status = "NEEDS_CLARIFICATION"
        reason = "no groundable concepts were supplied"
    else:
        status = decisions.status
        reason = decisions.reason
    return GroundedInterpretation(
        status=status,
        resolved_measures=list(dict.fromkeys(measures)),
        resolved_dimensions=list(dict.fromkeys(dimensions)),
        resolved_time_columns=list(dict.fromkeys(times)),
        concepts=concepts,
        reason=reason,
    )


__all__ = ["ground_interpretation"]
