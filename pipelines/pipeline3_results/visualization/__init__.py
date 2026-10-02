"""Visualization Intelligence for Pipeline 3 — Results.

This package provides deterministic, auditable column classification (Stage 1),
visualization feasibility analysis (Stage 2), candidate ranking (Stage 3),
and chart data generation (Stage 4) for tabular datasets.
"""

from __future__ import annotations

from pipelines.pipeline3_results.visualization.column_classifier import (
    classify_column,
    classify_columns,
    classify_dataframe,
)
from pipelines.pipeline3_results.visualization.data_builder import (
    build_chart_data,
)
from pipelines.pipeline3_results.visualization.explanation import (
    build_visualization_explanation,
)
from pipelines.pipeline3_results.visualization.feasibility import (
    analyze_visualization_feasibility,
)
from pipelines.pipeline3_results.visualization.integration import (
    build_visualization_payload,
)
from pipelines.pipeline3_results.visualization.orchestrator import (
    build_visualization_result,
)
from pipelines.pipeline3_results.visualization.ranker import (
    rank_visualization_candidates,
)
from pipelines.pipeline3_results.visualization.request_parser import (
    parse_visualization_request,
)
from pipelines.pipeline3_results.visualization.request_resolver import (
    resolve_visualization_request,
)
from pipelines.pipeline3_results.visualization.request_validator import (
    validate_visualization_request,
)
from pipelines.pipeline3_results.visualization.response_builder import (
    SUPPORTED_MODES,
    build_visualization_response,
)
from pipelines.pipeline3_results.visualization.results_service import (
    build_visualization_results,
)
from pipelines.pipeline3_results.visualization.schemas import (
    ALL_INTENTS,
    ALL_REQUEST_AGGREGATIONS,
    ALL_ROLES,
    ALL_VIZ_TYPES,
    CHART_TYPE_PRIORITY,
    STAGE4_MAX_CATEGORY_POINTS,
    STAGE4_MAX_HISTOGRAM_BINS,
    STAGE4_MAX_LINE_POINTS,
    STAGE4_MAX_SCATTER_POINTS,
    STAGE4_MAX_PIE_CATEGORIES,
    AggregationMethod,
    ChartData,
    ChartPresentationVariant,
    ChartPoint,
    ColumnClassification,
    ColumnRole,
    DatasetClassification,
    FeasibilityResult,
    HistogramBin,
    RankedCandidate,
    RankingResult,
    RequestAggregation,
    ValidatedVisualizationRequest,
    VisualizationCandidate,
    VisualizationIntent,
    VisualizationItem,
    VisualizationRequest,
    VisualizationRequestStatus,
    VisualizationResult,
    VisualizationStatus,
    VisualizationType,
)

__all__ = [
    # Stage 1
    "classify_column",
    "classify_columns",
    "classify_dataframe",
    "ALL_ROLES",
    "ColumnClassification",
    "ColumnRole",
    "DatasetClassification",
    # Stage 2
    "analyze_visualization_feasibility",
    "ALL_VIZ_TYPES",
    "FeasibilityResult",
    "VisualizationCandidate",
    "VisualizationType",
    # Stage 3
    "rank_visualization_candidates",
    "CHART_TYPE_PRIORITY",
    "RankedCandidate",
    "RankingResult",
    # Stage 4
    "build_chart_data",
    "AggregationMethod",
    "ChartData",
    "ChartPresentationVariant",
    "ChartPoint",
    "HistogramBin",
    "STAGE4_MAX_CATEGORY_POINTS",
    "STAGE4_MAX_LINE_POINTS",
    "STAGE4_MAX_SCATTER_POINTS",
    "STAGE4_MAX_HISTOGRAM_BINS",
    "STAGE4_MAX_PIE_CATEGORIES",
    # Stage 5
    "build_visualization_result",
    "VisualizationItem",
    "VisualizationResult",
    "VisualizationStatus",
    # Stage 6
    "build_visualization_payload",
    # Stage 7
    "parse_visualization_request",
    "validate_visualization_request",
    "VisualizationRequest",
    "ValidatedVisualizationRequest",
    "VisualizationIntent",
    "RequestAggregation",
    "VisualizationRequestStatus",
    "ALL_INTENTS",
    "ALL_REQUEST_AGGREGATIONS",
    # Stage 8
    "resolve_visualization_request",
    # Stage 9
    "build_visualization_explanation",
    # Stage 10
    "build_visualization_response",
    "build_visualization_results",
    "SUPPORTED_MODES",
]
