"""Metadata-driven semantic grounding with hard analytical-role gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from pipelines.pipeline3_results.grounding.schemas import (
    AnalyticalExpression,
    DatasetColumnKnowledge,
    DatasetKnowledge,
    GroundedConcept,
    GroundingCandidate,
    GroundingResult,
)
from pipelines.pipeline3_results.grounding.similarity import (
    LexicalSemanticSimilarityProvider,
    SemanticSimilarityProvider,
    semantic_tokens,
)
if TYPE_CHECKING:
    from pipelines.pipeline3_results.nlp.semantic_intent import SemanticIntent


_NUMERIC_SEMANTIC_TYPES = {"numeric_measurement", "currency", "percentage"}
_NUMERIC_ROLES = {"measurement"}
_NUMERIC_VALUE_FUNCTIONS = {"measurement"}
_NUMERIC_MEANINGS = {"magnitude"}
_TEMPORAL_TYPES = {"date", "time", "datetime"}
_TEMPORAL_ROLES = {"timestamp"}
_TEMPORAL_FUNCTIONS = {"temporal"}
_CATEGORY_TYPES = {"categorical", "boolean", "location"}
_CATEGORY_ROLES = {"category"}
_CATEGORY_FUNCTIONS = {"classification", "boolean_state", "location"}
_IDENTIFIER_TYPES = {"identifier"}
_IDENTIFIER_ROLES = {
    "primary_identifier", "foreign_identifier", "transaction_identifier"
}
_IDENTIFIER_FUNCTIONS = {"entity_reference"}
_COUNT_WORDS = {
    "count", "number", "many", "volume", "frequency", "frequent", "frequently",
    "total", "distinct", "unique", "how",
}
_ROW_WORDS = {"row", "record", "observation", "entry"}
_NON_NULL_WORDS = {"nonnull", "non", "null", "present", "filled", "available"}
_VARIATION_WORDS = {"variation", "variability", "spread", "dispersion"}
_STDDEV_WORDS = {"stddev", "standard", "deviation"}
_RANGE_WORDS = {"range"}
_GENERIC_TIME_WORDS = {"time", "over", "throughout", "period", "per", "by", "in"}
_GRANULARITIES = {
    "year": "YEAR", "yearly": "YEAR", "annual": "YEAR", "annually": "YEAR",
    "quarter": "QUARTER", "quarterly": "QUARTER",
    "month": "MONTH", "monthly": "MONTH",
    "week": "WEEK", "weekly": "WEEK",
    "day": "DAY", "daily": "DAY",
    "hour": "HOUR", "hourly": "HOUR",
}
_MATCH_THRESHOLD = 0.56
_AMBIGUITY_MARGIN = 0.08


@dataclass(frozen=True)
class _Compatibility:
    role: bool
    dtype: bool
    evidence: list[str]


def _fold(value: str | None) -> str:
    return (value or "").casefold()


def _has_semantic_classification(column: DatasetColumnKnowledge) -> bool:
    unknown = {"", "unknown", "not_applicable", "uncertain"}
    return any(
        _fold(value) not in unknown
        for value in (
            column.semantic_type,
            column.semantic_role,
            column.value_function,
            column.numeric_meaning,
        )
    )


def _is_numeric(column: DatasetColumnKnowledge) -> bool:
    dtype = _fold(column.physical_dtype)
    semantic_match = (
        _fold(column.semantic_type) in _NUMERIC_SEMANTIC_TYPES
        or _fold(column.semantic_role) in _NUMERIC_ROLES
        or _fold(column.value_function) in _NUMERIC_VALUE_FUNCTIONS
        or _fold(column.numeric_meaning) in _NUMERIC_MEANINGS
    )
    if semantic_match or _has_semantic_classification(column):
        return semantic_match
    return any(token in dtype for token in ("int", "float", "double", "decimal", "number"))


def _is_temporal(column: DatasetColumnKnowledge) -> bool:
    dtype = _fold(column.physical_dtype)
    semantic_match = (
        _fold(column.semantic_type) in _TEMPORAL_TYPES
        or _fold(column.semantic_role) in _TEMPORAL_ROLES
        or _fold(column.value_function) in _TEMPORAL_FUNCTIONS
    )
    if semantic_match or _has_semantic_classification(column):
        return semantic_match
    return "date" in dtype or "time" in dtype


def _is_identifier(column: DatasetColumnKnowledge) -> bool:
    return (
        _fold(column.semantic_type) in _IDENTIFIER_TYPES
        or _fold(column.semantic_role) in _IDENTIFIER_ROLES
        or _fold(column.value_function) in _IDENTIFIER_FUNCTIONS
        or _fold(column.numeric_meaning) == "identifier"
    )


def _is_category(column: DatasetColumnKnowledge) -> bool:
    dtype = _fold(column.physical_dtype)
    semantic_match = (
        _fold(column.semantic_type) in _CATEGORY_TYPES
        or _fold(column.semantic_role) in _CATEGORY_ROLES
        or _fold(column.value_function) in _CATEGORY_FUNCTIONS
    )
    if semantic_match or _has_semantic_classification(column):
        return semantic_match and not _is_temporal(column)
    return any(token in dtype for token in ("object", "string", "str", "category", "bool"))


def _compatibility(
    column: DatasetColumnKnowledge,
    slot: str,
    *,
    exact_name: bool = False,
) -> _Compatibility:
    numeric = _is_numeric(column)
    temporal = _is_temporal(column)
    identifier = _is_identifier(column)
    category = _is_category(column)
    if slot == "MEASURE":
        role_ok = numeric and not temporal and not identifier
        dtype_ok = numeric and not temporal
        expected = "numeric measurement"
    elif slot == "TIME":
        role_ok = temporal
        dtype_ok = temporal
        expected = "temporal field"
    elif slot == "DIMENSION":
        role_ok = category or temporal or (identifier and exact_name)
        dtype_ok = not (numeric and not identifier and not temporal)
        expected = "category, temporal field, or explicitly requested identifier"
    else:  # FILTER
        role_ok = category or temporal or numeric or identifier
        dtype_ok = True
        expected = "filterable field"
    evidence = [
        (
            f"role/type eligible as {expected}"
            if role_ok and dtype_ok
            else f"role/type ineligible; expected {expected}"
        )
    ]
    return _Compatibility(role_ok, dtype_ok, evidence)


def _column_text(column: DatasetColumnKnowledge, knowledge: DatasetKnowledge) -> str:
    values = [
        column.name,
        column.concept_description,
        column.semantic_type,
        column.semantic_role,
        column.value_function,
        column.numeric_meaning,
        column.referenced_entity,
        column.reasoning_summary,
    ]
    return " ".join(str(value) for value in values if value)


def _normal_expression(
    semantic: SemanticIntent,
    column: DatasetColumnKnowledge,
    slot: str,
) -> AnalyticalExpression:
    if slot == "MEASURE" and semantic.aggregation != "NONE":
        return AnalyticalExpression(
            kind="AGGREGATION",
            operation=semantic.aggregation,  # type: ignore[arg-type]
            column=column.name,
        )
    return AnalyticalExpression(kind="COLUMN", column=column.name)


def _candidate(
    concept: str,
    column: DatasetColumnKnowledge,
    knowledge: DatasetKnowledge,
    slot: str,
    provider: SemanticSimilarityProvider,
    expression: AnalyticalExpression,
) -> GroundingCandidate:
    concept_normalized = " ".join(semantic_tokens(concept))
    name_normalized = " ".join(semantic_tokens(column.name))
    exact = bool(concept_normalized) and concept_normalized == name_normalized
    similarity = 1.0 if exact else provider.similarity(
        concept, _column_text(column, knowledge)
    )
    compatibility = _compatibility(column, slot, exact_name=exact)
    eligible = compatibility.role and compatibility.dtype
    if not eligible:
        tier = "INELIGIBLE"
    elif exact:
        tier = "EXACT"
    elif similarity >= _MATCH_THRESHOLD:
        tier = "SEMANTIC"
    else:
        tier = "WEAK"
    evidence = [
        *( ["exact normalized column-name match"] if exact else [] ),
        f"metadata semantic similarity={similarity:.3f}",
        *compatibility.evidence,
    ]
    if column.concept_description:
        evidence.append(f"concept_description: {column.concept_description}")
    if column.semantic_role:
        evidence.append(f"semantic_role: {column.semantic_role}")
    if column.semantic_type:
        evidence.append(f"semantic_type: {column.semantic_type}")
    if column.value_function:
        evidence.append(f"value_function: {column.value_function}")
    return GroundingCandidate(
        expression=expression,
        semantic_score=similarity,
        role_compatible=compatibility.role,
        type_compatible=compatibility.dtype,
        evidence_tier=tier,  # type: ignore[arg-type]
        evidence=evidence,
        pipeline1_confidence=column.confidence,
    )


def _choose(
    concept: str,
    slot: str,
    candidates: list[GroundingCandidate],
    *,
    broad: bool = False,
) -> GroundedConcept:
    ordered = sorted(
        candidates,
        key=lambda item: (
            not (item.role_compatible and item.type_compatible),
            -item.semantic_score,
            item.expression.column or "",
            item.expression.kind,
            item.expression.operation or "",
        ),
    )
    eligible = [
        item for item in ordered
        if item.role_compatible and item.type_compatible
        and (broad or item.semantic_score >= _MATCH_THRESHOLD)
    ]
    if not eligible:
        return GroundedConcept(
            source_concept=concept,
            slot_type=slot,  # type: ignore[arg-type]
            status="UNRESOLVED",
            candidates=ordered,
            clarification_reason="No role-compatible semantic candidate met the grounding threshold.",
        )
    if broad and len(eligible) > 1:
        return GroundedConcept(
            source_concept=concept,
            slot_type=slot,  # type: ignore[arg-type]
            status="AMBIGUOUS",
            candidates=eligible,
            clarification_reason="The semantic interpreter marked this supplied concept as broad.",
        )
    top = eligible[0]
    competing = [
        item for item in eligible[1:]
        if top.semantic_score - item.semantic_score <= _AMBIGUITY_MARGIN
    ]
    if competing:
        return GroundedConcept(
            source_concept=concept,
            slot_type=slot,  # type: ignore[arg-type]
            status="AMBIGUOUS",
            candidates=[top, *competing],
            clarification_reason="Multiple role-compatible candidates have comparable semantic evidence.",
        )
    return GroundedConcept(
        source_concept=concept,
        slot_type=slot,  # type: ignore[arg-type]
        status="RESOLVED",
        resolved_expression=top.expression,
        candidates=ordered,
        evidence=list(top.evidence),
    )


def _ground_regular(
    concept: str,
    slot: str,
    semantic: SemanticIntent,
    knowledge: DatasetKnowledge,
    provider: SemanticSimilarityProvider,
    *,
    broad: bool = False,
) -> GroundedConcept:
    candidates = [
        _candidate(
            concept,
            column,
            knowledge,
            slot,
            provider,
            _normal_expression(semantic, column, slot),
        )
        for column in knowledge.columns
    ]
    return _choose(concept, slot, candidates, broad=broad)


def _count_core(concept: str) -> str:
    return " ".join(
        token for token in semantic_tokens(concept)
        if token not in _COUNT_WORDS | _NON_NULL_WORDS
    )


def _ground_count(
    concept: str,
    semantic: SemanticIntent,
    knowledge: DatasetKnowledge,
    provider: SemanticSimilarityProvider,
) -> GroundedConcept:
    tokens = set(semantic_tokens(concept))
    if tokens and tokens.issubset(_ROW_WORDS | _COUNT_WORDS) and tokens & _ROW_WORDS:
        expression = AnalyticalExpression(kind="COUNT_ROWS")
        candidate = GroundingCandidate(
            expression=expression,
            semantic_score=1.0,
            role_compatible=True,
            type_compatible=True,
            evidence_tier="EXACT",
            evidence=["request explicitly counts rows/records/observations"],
        )
        return GroundedConcept(
            source_concept=concept,
            slot_type="MEASURE",
            status="RESOLVED",
            resolved_expression=expression,
            candidates=[candidate],
            evidence=list(candidate.evidence),
        )

    core = _count_core(concept) or concept
    explicit_non_null = (
        "nonnull" in tokens
        or {"non", "null"}.issubset(tokens)
        or bool(tokens & {"present", "filled", "available"})
    )
    if explicit_non_null:
        candidates = [
            _candidate(
                core,
                column,
                knowledge,
                "FILTER",
                provider,
                AnalyticalExpression(kind="COUNT_NON_NULL", column=column.name),
            )
            for column in knowledge.columns
        ]
        return _choose(concept, "MEASURE", candidates)

    candidates: list[GroundingCandidate] = []
    for column in knowledge.columns:
        expression = AnalyticalExpression(
            kind="COUNT_DISTINCT", operation="COUNT_DISTINCT", column=column.name
        )
        base = _candidate(core, column, knowledge, "FILTER", provider, expression)
        identifier = _is_identifier(column)
        evidence = [*base.evidence, (
            "identifier/entity semantics permit distinct counting"
            if identifier else "not an entity identifier"
        )]
        candidates.append(base.model_copy(update={
            "role_compatible": identifier,
            "type_compatible": identifier,
            "evidence_tier": base.evidence_tier if identifier else "INELIGIBLE",
            "evidence": evidence,
        }))
    selected = _choose(concept, "MEASURE", candidates)
    if selected.status != "UNRESOLVED":
        return selected

    row_score = provider.similarity(core, knowledge.row_entity or "")
    if row_score >= _MATCH_THRESHOLD or semantic.dimension_concepts:
        expression = AnalyticalExpression(kind="COUNT_ROWS")
        row_candidate = GroundingCandidate(
            expression=expression,
            semantic_score=max(row_score, 0.65 if semantic.dimension_concepts else 0.0),
            role_compatible=True,
            type_compatible=True,
            evidence_tier="SEMANTIC",
            evidence=[
                "no matching entity identifier is available",
                "row frequency is supported by row-entity or grouping context",
            ],
        )
        return GroundedConcept(
            source_concept=concept,
            slot_type="MEASURE",
            status="RESOLVED",
            resolved_expression=expression,
            candidates=[*candidates, row_candidate],
            evidence=list(row_candidate.evidence),
        )
    return selected


def _variation_operation(tokens: set[str]) -> str | None:
    if "stddev" in tokens or {"standard", "deviation"}.issubset(tokens):
        return "STDDEV"
    if tokens & _RANGE_WORDS:
        return "RANGE"
    return None


def _ground_variation(
    concept: str,
    semantic: SemanticIntent,
    knowledge: DatasetKnowledge,
    provider: SemanticSimilarityProvider,
) -> GroundedConcept:
    tokens = set(semantic_tokens(concept))
    base = " ".join(
        token for token in semantic_tokens(concept)
        if token not in _VARIATION_WORDS | _STDDEV_WORDS | _RANGE_WORDS
    ) or concept
    explicit_operation = _variation_operation(tokens)
    operations = [explicit_operation] if explicit_operation else ["STDDEV", "RANGE"]
    candidates: list[GroundingCandidate] = []
    for column in knowledge.columns:
        for operation in operations:
            expression = AnalyticalExpression(
                kind="STATISTIC", operation=operation, column=column.name  # type: ignore[arg-type]
            )
            candidates.append(_candidate(
                base, column, knowledge, "MEASURE", provider, expression
            ))
    result = _choose(concept, "MEASURE", candidates)
    if explicit_operation or result.status != "RESOLVED":
        return result
    # A generic variation concept deliberately preserves both valid statistics.
    top_column = result.resolved_expression.column if result.resolved_expression else None
    alternatives = [
        item for item in candidates
        if item.expression.column == top_column
        and item.role_compatible and item.type_compatible
        and item.semantic_score >= _MATCH_THRESHOLD
    ]
    return GroundedConcept(
        source_concept=concept,
        slot_type="MEASURE",
        status="AMBIGUOUS",
        candidates=alternatives,
        clarification_reason="Variation may mean standard deviation or range; choose one statistic.",
    )


def _time_granularity(expression: str) -> str | None:
    for token in semantic_tokens(expression):
        if token in _GRANULARITIES:
            return _GRANULARITIES[token]
    return None


def _ground_time(
    expression: str,
    knowledge: DatasetKnowledge,
    provider: SemanticSimilarityProvider,
) -> GroundedConcept:
    tokens = set(semantic_tokens(expression))
    granularity = _time_granularity(expression)
    generic = bool(tokens) and tokens.issubset(
        _GENERIC_TIME_WORDS | set(_GRANULARITIES)
    )
    candidates: list[GroundingCandidate] = []
    for column in knowledge.columns:
        analytical = (
            AnalyticalExpression(
                kind="TIME_BUCKET", column=column.name,
                time_granularity=granularity,  # type: ignore[arg-type]
            )
            if granularity else AnalyticalExpression(kind="COLUMN", column=column.name)
        )
        candidate = _candidate(
            expression, column, knowledge, "TIME", provider, analytical
        )
        if generic and candidate.role_compatible and candidate.type_compatible:
            candidate = candidate.model_copy(update={
                "semantic_score": 0.7,
                "evidence_tier": "SEMANTIC",
                "evidence": [*candidate.evidence, "generic temporal expression"],
            })
        candidates.append(candidate)
    return _choose(expression, "TIME", candidates)


def _declared_broad(semantic: SemanticIntent, slot: str, concept: str) -> bool:
    return any(
        item.slot == slot and item.concept.casefold() == concept.casefold()
        for item in semantic.ambiguities
    )


def _collect(items: Iterable[GroundedConcept], status: str) -> list[GroundedConcept]:
    return [item for item in items if item.status == status]


def ground_semantic_intent(
    semantic_intent: SemanticIntent,
    dataset_knowledge: DatasetKnowledge,
    similarity_provider: SemanticSimilarityProvider | None = None,
) -> GroundingResult:
    """Ground all semantic slots using metadata only; never inspect raw rows."""
    from pipelines.pipeline3_results.nlp.semantic_intent import SemanticIntent

    if not isinstance(semantic_intent, SemanticIntent):
        raise TypeError("semantic_intent must be a SemanticIntent")
    if not isinstance(dataset_knowledge, DatasetKnowledge):
        raise TypeError("dataset_knowledge must be DatasetKnowledge")
    provider = similarity_provider or LexicalSemanticSimilarityProvider()

    if semantic_intent.unsupported:
        return GroundingResult(
            status="UNSUPPORTED",
            signals=["grounding:semantic_intent_unsupported"],
        )

    measures: list[GroundedConcept] = []
    for concept in semantic_intent.measure_concepts:
        tokens = set(semantic_tokens(concept))
        if semantic_intent.analytical_goal == "COUNT_ENTITIES" or semantic_intent.aggregation == "COUNT":
            grounded = _ground_count(concept, semantic_intent, dataset_knowledge, provider)
        elif tokens & (_VARIATION_WORDS | _STDDEV_WORDS | _RANGE_WORDS):
            grounded = _ground_variation(
                concept, semantic_intent, dataset_knowledge, provider
            )
        else:
            grounded = _ground_regular(
                concept,
                "MEASURE",
                semantic_intent,
                dataset_knowledge,
                provider,
                broad=_declared_broad(semantic_intent, "measure", concept),
            )
        measures.append(grounded)

    dimensions = [
        _ground_regular(
            concept,
            "DIMENSION",
            semantic_intent,
            dataset_knowledge,
            provider,
            broad=_declared_broad(semantic_intent, "dimension", concept),
        )
        for concept in semantic_intent.dimension_concepts
    ]
    filters = [
        _ground_regular(
            item.field_concept,
            "FILTER",
            semantic_intent,
            dataset_knowledge,
            provider,
        )
        for item in semantic_intent.filters
    ]
    temporal = (
        [_ground_time(semantic_intent.time_expression, dataset_knowledge, provider)]
        if semantic_intent.time_expression else []
    )

    for missing in semantic_intent.missing_slots:
        slot = {
            "measure": "MEASURE",
            "aggregation": "MEASURE",
            "dimension": "DIMENSION",
            "comparison": "DIMENSION",
            "time_expression": "TIME",
        }[missing.slot]
        item = GroundedConcept(
            source_concept=missing.slot,
            slot_type=slot,  # type: ignore[arg-type]
            status="UNRESOLVED",
            clarification_reason=missing.reason,
        )
        if slot == "MEASURE":
            measures.append(item)
        elif slot == "DIMENSION":
            dimensions.append(item)
        else:
            temporal.append(item)

    all_items = [*measures, *dimensions, *filters, *temporal]
    ambiguities = _collect(all_items, "AMBIGUOUS")
    unresolved = _collect(all_items, "UNRESOLVED")
    if unresolved:
        status = "UNRESOLVED"
    elif ambiguities:
        status = "AMBIGUOUS"
    else:
        status = "RESOLVED"
    return GroundingResult(
        status=status,  # type: ignore[arg-type]
        measures=measures,
        dimensions=dimensions,
        filters=filters,
        temporal=temporal,
        ambiguities=ambiguities,
        unresolved=unresolved,
        signals=[
            "grounding:dataset_knowledge",
            f"grounding:similarity_provider:{type(provider).__name__}",
            f"grounding:{status.casefold()}",
        ],
    )


is_numeric_eligible = _is_numeric
is_temporal_eligible = _is_temporal
is_identifier_eligible = _is_identifier
is_category_eligible = _is_category


__all__ = [
    "ground_semantic_intent",
    "is_category_eligible",
    "is_identifier_eligible",
    "is_numeric_eligible",
    "is_temporal_eligible",
]
