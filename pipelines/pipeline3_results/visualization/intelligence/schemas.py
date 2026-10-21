"""Frozen renderer-neutral visualization contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ChartFamily(StrEnum):
    BAR = "BAR"
    GROUPED_BAR = "GROUPED_BAR"
    STACKED_BAR = "STACKED_BAR"
    LINE = "LINE"
    AREA = "AREA"
    PIE = "PIE"
    DOUGHNUT = "DOUGHNUT"
    SCATTER = "SCATTER"
    HISTOGRAM = "HISTOGRAM"
    BOX_PLOT = "BOX_PLOT"
    HEATMAP = "HEATMAP"
    TREEMAP = "TREEMAP"
    SUNBURST = "SUNBURST"
    FUNNEL = "FUNNEL"
    RADAR = "RADAR"
    GAUGE = "GAUGE"
    SANKEY = "SANKEY"
    COMBINATION = "COMBINATION"


class VisualizationPurpose(StrEnum):
    COMPARISON = "COMPARISON"
    RANKING = "RANKING"
    TREND = "TREND"
    DISTRIBUTION = "DISTRIBUTION"
    RELATIONSHIP = "RELATIONSHIP"
    PART_TO_WHOLE = "PART_TO_WHOLE"
    COMPOSITION = "COMPOSITION"
    DEVIATION = "DEVIATION"
    FLOW = "FLOW"
    HIERARCHY = "HIERARCHY"
    KPI_SUPPORT = "KPI_SUPPORT"


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class FieldEncoding(StrictFrozenModel):
    channel: Literal["X", "Y", "COLOR", "SIZE", "ANGLE", "SERIES", "CATEGORY", "VALUE"]
    field: str
    semantic_label: str
    data_type: str
    role: str
    source_field: str | None = None
    format_hint: Literal["NUMBER", "INTEGER", "PERCENT", "CURRENCY", "DURATION", "DATE", "DATETIME"] = "NUMBER"


class ChartSeries(StrictFrozenModel):
    id: str
    label: str
    measure_field: str
    dimension_filter: dict[str, JsonValue] | None = None
    stack_group: str | None = None


class SuitabilityAssessment(StrictFrozenModel):
    status: Literal["SUITABLE", "SUITABLE_WITH_WARNING", "NOT_RECOMMENDED", "INVALID"]
    reason_codes: list[str]
    message: str


class VisualizationPlan(StrictFrozenModel):
    visualization_id: str
    source: Literal["AUTO", "USER_REQUESTED"]
    analytical_plan_id: str
    verified_result_id: str
    chart_family: ChartFamily
    purpose: VisualizationPurpose
    x_encoding: FieldEncoding | None = None
    y_encoding: FieldEncoding | None = None
    series_encoding: FieldEncoding | None = None
    aggregation_context: str | None = None
    sort_context: str | None = None
    temporal_context: str | None = None
    suitability: SuitabilityAssessment
    why_selected: list[str]
    warnings: list[str] = Field(default_factory=list)


class ChartData(StrictFrozenModel):
    columns: list[str]
    rows: list[dict[str, JsonValue]]
    coverage: dict[str, int]
    truncated: bool
    display_limit: int | None = None
    other_bucket_allowed: bool = False
    source_result_id: str


class RendererNeutralChartSpec(StrictFrozenModel):
    chart_id: str
    family: ChartFamily
    title: str
    subtitle: str | None = None
    encodings: list[FieldEncoding]
    series: list[ChartSeries] = Field(default_factory=list)
    orientation: Literal["HORIZONTAL", "VERTICAL", "RADIAL", "CARTESIAN"]
    stacking: Literal["NONE", "NORMAL", "PERCENT"] = "NONE"
    legend: Literal["AUTO", "VISIBLE", "HIDDEN"] = "AUTO"
    tooltip: bool = True
    labels: Literal["AUTO", "NONE", "VALUE", "PERCENT", "CATEGORY_AND_VALUE"] = "AUTO"
    number_format: Literal["NUMBER", "INTEGER", "PERCENT", "CURRENCY", "DURATION", "DATE", "DATETIME"] = "NUMBER"
    palette_role: Literal["CATEGORICAL", "SEQUENTIAL", "DIVERGING", "SINGLE_SERIES", "MULTI_SERIES", "STATUS"]
    interaction_hints: list[Literal["ZOOM", "PAN", "HOVER", "LEGEND_TOGGLE", "DATA_VIEW", "EXPORT"]]
    data_ref: ChartData
    warnings: list[str] = Field(default_factory=list)


class ScoreComponent(StrictFrozenModel):
    name: str
    value: float = Field(ge=-1.0, le=1.0)
    reason_code: str


class DashboardVisualization(StrictFrozenModel):
    id: str
    source: Literal["AUTO", "USER_REQUESTED"]
    title: str
    why_selected: list[str]
    analytical_plan_ref: str
    verified_result_ref: str
    score: float | None = None
    score_components: list[ScoreComponent] = Field(default_factory=list)
    plan: VisualizationPlan
    chart_spec: RendererNeutralChartSpec


class KPICandidate(StrictFrozenModel):
    id: str
    label: str
    expression: dict[str, JsonValue]
    verified_result_id: str
    value: JsonValue
    format_hint: Literal["NUMBER", "INTEGER", "PERCENT", "CURRENCY", "DURATION"]
    importance: float = Field(ge=0.0, le=1.0)
    evidence: list[str]


class VisualizationDashboard(StrictFrozenModel):
    dataset_id: str
    status: Literal["READY", "EMPTY", "PARTIAL"]
    kpis: list[KPICandidate]
    visualizations: list[DashboardVisualization]
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "ChartData", "ChartFamily", "ChartSeries", "DashboardVisualization",
    "FieldEncoding", "KPICandidate", "RendererNeutralChartSpec", "ScoreComponent",
    "SuitabilityAssessment", "VisualizationDashboard", "VisualizationPlan",
    "VisualizationPurpose",
]
