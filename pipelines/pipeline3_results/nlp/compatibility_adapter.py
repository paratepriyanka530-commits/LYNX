"""Temporary SemanticIntent adapter for existing Pipeline 3 consumers."""

from __future__ import annotations

import re
from typing import Any, Mapping

import pandas as pd

from pipelines.pipeline3_results.grounding.compatibility import adapt_grounding_result
from pipelines.pipeline3_results.grounding.dataset_knowledge import build_dataset_knowledge
from pipelines.pipeline3_results.grounding.engine import ground_semantic_intent
from pipelines.pipeline3_results.planning.planner import build_analytical_plan
from pipelines.pipeline3_results.nlp.ambiguity_detector import detect_ambiguities
from pipelines.pipeline3_results.nlp.column_resolver import resolve_columns
from pipelines.pipeline3_results.nlp.normalizer import normalize_text
from pipelines.pipeline3_results.nlp.schemas import (
    AmbiguityDecision,
    ExtractedEntities,
    QueryFilter,
    QueryInterpretation,
    StructuredQuery,
)
from pipelines.pipeline3_results.nlp.semantic_intent import SemanticIntent
from pipelines.pipeline3_results.visualization.schemas import DatasetClassification


_FILTER_OPERATORS = {
    "EQUALS", "NOT_EQUALS", "GREATER_THAN", "GREATER_THAN_OR_EQUAL",
    "LESS_THAN", "LESS_THAN_OR_EQUAL", "CONTAINS", "UNKNOWN",
}


def _kind(slot: str) -> str:
    return {
        "measure": "MEASURE",
        "dimension": "DIMENSION",
        "time_expression": "TIME",
        "aggregation": "MEASURE",
        "comparison": "DIMENSION",
    }[slot]


def _chart(intent: SemanticIntent) -> tuple[str | None, str | None]:
    if intent.requested_chart == "DOUGHNUT":
        return "PIE", "DOUGHNUT"
    return intent.requested_chart, None


def _adapt_with_legacy_resolver(
    semantic: SemanticIntent,
    raw_text: str,
    dataframe: pd.DataFrame,
    *,
    pipeline1_report: Mapping[str, Any] | None = None,
    classification: DatasetClassification | None = None,
) -> QueryInterpretation:
    """Ground through the current resolver and emit the legacy query contract."""
    if not isinstance(semantic, SemanticIntent):
        raise TypeError("semantic must be a SemanticIntent")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    requested_chart, chart_variant = _chart(semantic)
    filters = [
        QueryFilter(
            raw_text=" ".join(
                part for part in (
                    item.field_concept, item.operator, item.value_concept
                ) if part
            ),
            column_term=item.field_concept,
            operator=(item.operator if item.operator in _FILTER_OPERATORS else "UNKNOWN"),  # type: ignore[arg-type]
            value_term=item.value_concept,
        )
        for item in semantic.filters
    ]
    entities = ExtractedEntities(
        measure_terms=list(semantic.measure_concepts),
        dimension_terms=list(semantic.dimension_concepts),
        aggregation=semantic.aggregation,
        comparison=semantic.comparison,
        requested_chart=requested_chart,  # type: ignore[arg-type]
        requested_chart_variant=chart_variant,  # type: ignore[arg-type]
        filters=filters,
        time_terms=[semantic.time_expression] if semantic.time_expression else [],
        limit=semantic.limit,
        signals=("pretrained_semantic_intent",),
    )

    base_signals = [
        "semantic_interpreter:VALID_INTENT",
        "semantic_interpreter:source:LLM",
        f"semantic_interpreter:goal:{semantic.analytical_goal}",
    ]
    if semantic.unsupported:
        query = _empty_query(raw_text, semantic, entities)
        return QueryInterpretation(
            status="NEEDS_CLARIFICATION",
            query=query,
            intent_confidence="LOW",
            message=semantic.unsupported_reason,
            signals=[*base_signals, "semantic_interpreter:unsupported"],
        )

    semantic_decisions = [
        AmbiguityDecision(
            term=item.slot,
            kind=_kind(item.slot),  # type: ignore[arg-type]
            status="NEEDS_CLARIFICATION",
            reason=item.reason,
            signals=("semantic_intent_missing_slot",),
        )
        for item in semantic.missing_slots
    ] + [
        AmbiguityDecision(
            term=item.concept,
            kind=_kind(item.slot),  # type: ignore[arg-type]
            status="NEEDS_CLARIFICATION",
            reason=item.reason,
            signals=("semantic_intent_ambiguity",),
        )
        for item in semantic.ambiguities
    ]
    if semantic.needs_clarification:
        query = _empty_query(raw_text, semantic, entities)
        return QueryInterpretation(
            status="NEEDS_CLARIFICATION",
            query=query,
            ambiguity_decisions=semantic_decisions,
            intent_confidence="LOW",
            message=semantic.clarification_question,
            signals=[*base_signals, "semantic_interpreter:clarification_required"],
        )

    resolutions = resolve_columns(
        entities,
        dataframe,
        pipeline1_report=pipeline1_report,
        classification=classification,
    )
    if semantic.analytical_goal == "RELATIONSHIP":
        resolutions = [
            resolution.model_copy(update={"kind": "RELATIONSHIP"})
            if resolution.kind == "MEASURE" else resolution
            for resolution in resolutions
        ]
    ambiguity = detect_ambiguities(resolutions)
    resolved_measures: list[str] = []
    resolved_dimensions: list[str] = []
    resolved_times: list[str] = []
    attached = []
    for resolution, decision in zip(resolutions, ambiguity.decisions, strict=True):
        attached.append(resolution.model_copy(update={
            "resolved_column": decision.resolved_column,
            "reason": decision.reason,
        }))
        if decision.status != "RESOLVED" or decision.resolved_column is None:
            continue
        if decision.kind in {"MEASURE", "RELATIONSHIP"}:
            resolved_measures.append(decision.resolved_column)
        elif decision.kind == "DIMENSION":
            resolved_dimensions.append(decision.resolved_column)
        elif decision.kind == "TIME":
            resolved_times.append(decision.resolved_column)

    filter_decisions = [d for d in ambiguity.decisions if d.kind == "FILTER"]
    resolved_filters = [
        item.model_copy(update={
            "resolved_column": (
                filter_decisions[index].resolved_column
                if index < len(filter_decisions) else None
            )
        })
        for index, item in enumerate(filters)
    ]
    query_time_terms = list(entities.time_terms)
    if (
        semantic.analytical_goal == "TREND"
        and len(query_time_terms) == 1
        and not re.search(
            r"\d|\b(?:last|this|next|before|after|between)\b",
            query_time_terms[0],
            re.IGNORECASE,
        )
    ):
        # Current visualization compatibility treats "over time" as an axis
        # request rather than an executable temporal restriction. Resolution
        # above still used the model's original semantic time expression.
        query_time_terms = ["over time"]
    query = StructuredQuery(
        raw_text=raw_text,
        normalized_text=normalize_text(raw_text),
        intent=semantic.intent,
        measure_terms=list(semantic.measure_concepts),
        dimension_terms=list(semantic.dimension_concepts),
        aggregation=semantic.aggregation,
        comparison=semantic.comparison,
        requested_chart=requested_chart,  # type: ignore[arg-type]
        requested_chart_variant=chart_variant,  # type: ignore[arg-type]
        filters=resolved_filters,
        time_terms=query_time_terms,
        limit=semantic.limit,
        resolved_measures=resolved_measures,
        resolved_dimensions=resolved_dimensions,
        resolved_time_columns=resolved_times,
        resolutions=attached,
    )
    choices = [
        choice
        for decision in ambiguity.decisions
        if decision.status == "NEEDS_CLARIFICATION"
        for choice in decision.choices
    ]
    status = ambiguity.status if resolutions else "NEEDS_CLARIFICATION"
    message = ambiguity.reason if resolutions else "no_analytical_entities"
    return QueryInterpretation(
        status=status,
        query=query,
        ambiguity_decisions=ambiguity.decisions,
        intent_confidence="LOW",
        message=message,
        choices=list(dict.fromkeys(choices)),
        signals=[
            *base_signals,
            (
                "semantic_interpreter:grounded_by_existing_resolver"
                if status == "RESOLVED"
                else "semantic_interpreter:existing_grounding_incomplete"
            ),
        ],
    )


def adapt_semantic_intent(
    semantic: SemanticIntent,
    raw_text: str,
    dataframe: pd.DataFrame,
    *,
    pipeline1_report: Mapping[str, Any] | None = None,
    classification: DatasetClassification | None = None,
    allow_legacy_fallback: bool = True,
) -> QueryInterpretation:
    """Prefer DatasetKnowledge grounding and label the temporary legacy fallback."""
    if not isinstance(semantic, SemanticIntent):
        raise TypeError("semantic must be a SemanticIntent")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    knowledge = build_dataset_knowledge(
        pipeline1_report,
        dataset_id=(classification.dataset_name if classification else None),
        dataframe_schema={str(name): str(dtype) for name, dtype in dataframe.dtypes.items()},
    )
    grounding = ground_semantic_intent(semantic, knowledge)
    if (
        allow_legacy_fallback
        and not knowledge.metadata_availability.get("column_semantics", False)
    ):
        legacy = _adapt_with_legacy_resolver(
            semantic,
            raw_text,
            dataframe,
            pipeline1_report=pipeline1_report,
            classification=classification,
        )
        return legacy.model_copy(update={
            "signals": [
                *legacy.signals,
                *grounding.signals,
                "grounding:source:LEGACY_FALLBACK",
                "grounding:legacy_fallback_reason:semantic_metadata_unavailable",
            ]
        })
    planning = build_analytical_plan(semantic, grounding, knowledge)
    compatibility = adapt_grounding_result(semantic, grounding, raw_text)
    planning_signals = [
        f"planner:{planning.status}",
        "planner:canonical_analytical_plan",
    ]
    if planning.plan is not None:
        planning_signals.append(f"analytical_plan:{planning.plan.plan_id}")
        return compatibility.model_copy(update={
            "signals": [*compatibility.signals, *planning_signals],
            "analytical_plan": planning.plan.model_dump(mode="json"),
        })
    status = (
        "INVALID" if planning.status == "INVALID" else "NEEDS_CLARIFICATION"
    )
    return compatibility.model_copy(update={
        "status": status,
        "message": planning.message,
        "choices": list(planning.clarification_options),
        "signals": [*compatibility.signals, *planning_signals],
    })


def _empty_query(
    raw_text: str,
    semantic: SemanticIntent,
    entities: ExtractedEntities,
) -> StructuredQuery:
    return StructuredQuery(
        raw_text=raw_text,
        normalized_text=normalize_text(raw_text),
        intent=semantic.intent,
        measure_terms=list(entities.measure_terms),
        dimension_terms=list(entities.dimension_terms),
        aggregation=entities.aggregation,
        comparison=entities.comparison,
        requested_chart=entities.requested_chart,
        requested_chart_variant=entities.requested_chart_variant,
        filters=list(entities.filters),
        time_terms=list(entities.time_terms),
        limit=entities.limit,
    )


__all__ = ["adapt_semantic_intent"]
