"""Shared deterministic natural-language contracts for Pipeline 3."""

from pipelines.pipeline3_results.nlp.ambiguity_detector import (
    detect_ambiguities,
    detect_ambiguity,
)
from pipelines.pipeline3_results.nlp.intent_classifier import classify_intent
from pipelines.pipeline3_results.nlp.entity_extractor import extract_entities
from pipelines.pipeline3_results.nlp.column_resolver import (
    resolve_column_terms,
    resolve_columns,
)
from pipelines.pipeline3_results.nlp.normalizer import (
    normalize_aggregation,
    normalize_chart,
    normalize_comparison,
    normalize_language,
    normalize_text,
    tokenize,
)
from pipelines.pipeline3_results.nlp.orchestrator import (
    interpret_query,
    orchestrate_query,
)
from pipelines.pipeline3_results.nlp.schemas import (
    ALL_AGGREGATIONS,
    ALL_COMPARISON_OPERATIONS,
    ALL_INTERPRETATION_STATUSES,
    ALL_QUERY_INTENTS,
    ALL_SHARED_CHART_TYPES,
    Aggregation,
    AmbiguityDecision,
    AmbiguityResult,
    ColumnCandidate,
    ComparisonOperation,
    ExtractedEntities,
    FilterOperator,
    IntentClassification,
    IntentConfidence,
    InterpretationStatus,
    QueryFilter,
    QueryIntent,
    QueryInterpretation,
    SharedChartType,
    StructuredQuery,
    TermKind,
    TermResolution,
)

__all__ = [
    "ALL_AGGREGATIONS",
    "ALL_COMPARISON_OPERATIONS",
    "ALL_INTERPRETATION_STATUSES",
    "ALL_QUERY_INTENTS",
    "ALL_SHARED_CHART_TYPES",
    "Aggregation",
    "AmbiguityDecision",
    "AmbiguityResult",
    "ColumnCandidate",
    "ComparisonOperation",
    "ExtractedEntities",
    "FilterOperator",
    "IntentClassification",
    "IntentConfidence",
    "InterpretationStatus",
    "QueryFilter",
    "QueryIntent",
    "QueryInterpretation",
    "SharedChartType",
    "StructuredQuery",
    "TermKind",
    "TermResolution",
    "normalize_aggregation",
    "normalize_chart",
    "normalize_comparison",
    "normalize_language",
    "normalize_text",
    "tokenize",
    "classify_intent",
    "detect_ambiguities",
    "detect_ambiguity",
    "extract_entities",
    "interpret_query",
    "orchestrate_query",
    "resolve_column_terms",
    "resolve_columns",
]
