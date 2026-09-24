"""Schemas for Pipeline 3 — Results (Visualization Intelligence).

This module defines deterministic, auditable data contracts for column
and dataset classification.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

ColumnRole = Literal[
    "MEASURE",
    "CATEGORY",
    "TEMPORAL",
    "IDENTIFIER",
    "DESCRIPTIVE_TEXT",
    "BOOLEAN",
    "UNKNOWN",
]

ALL_ROLES: tuple[ColumnRole, ...] = (
    "MEASURE",
    "CATEGORY",
    "TEMPORAL",
    "IDENTIFIER",
    "DESCRIPTIVE_TEXT",
    "BOOLEAN",
    "UNKNOWN",
)


class ColumnClassification(BaseModel):
    """Deterministic classification of an individual tabular column."""

    model_config = ConfigDict(frozen=True)

    name: str
    role: ColumnRole
    dtype: str
    unique_count: int
    unique_ratio: float
    missing_count: int
    missing_ratio: float
    signals: list[str] = Field(default_factory=list)


class DatasetClassification(BaseModel):
    """Dataset-level column classification summary."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str | None = None
    row_count: int
    column_count: int
    columns: list[ColumnClassification] = Field(default_factory=list)
    role_counts: dict[str, int] = Field(default_factory=dict)

    @property
    def role_map(self) -> dict[str, ColumnRole]:
        """Map of column name to assigned ColumnRole."""
        return {col.name: col.role for col in self.columns}

    def get_column(self, name: str) -> ColumnClassification | None:
        """Find a column classification by exact or case-insensitive name."""
        for col in self.columns:
            if col.name == name:
                return col
        normalized = name.casefold()
        for col in self.columns:
            if col.name.casefold() == normalized:
                return col
        return None

    def columns_by_role(self, role: ColumnRole) -> list[ColumnClassification]:
        """Return all columns assigned a specific role."""
        return [col for col in self.columns if col.role == role]

    @property
    def measures(self) -> list[str]:
        return [col.name for col in self.columns if col.role == "MEASURE"]

    @property
    def categories(self) -> list[str]:
        return [col.name for col in self.columns if col.role == "CATEGORY"]

    @property
    def temporals(self) -> list[str]:
        return [col.name for col in self.columns if col.role == "TEMPORAL"]

    @property
    def identifiers(self) -> list[str]:
        return [col.name for col in self.columns if col.role == "IDENTIFIER"]

    @property
    def descriptive_texts(self) -> list[str]:
        return [col.name for col in self.columns if col.role == "DESCRIPTIVE_TEXT"]

    @property
    def booleans(self) -> list[str]:
        return [col.name for col in self.columns if col.role == "BOOLEAN"]

    @property
    def unknowns(self) -> list[str]:
        return [col.name for col in self.columns if col.role == "UNKNOWN"]


# ---------------------------------------------------------------------------
# Stage 2: Visualization Feasibility Schemas
# ---------------------------------------------------------------------------

VisualizationType = Literal[
    "BAR",
    "LINE",
    "SCATTER",
    "HISTOGRAM",
    "COUNT_BAR",
]

ALL_VIZ_TYPES: tuple[VisualizationType, ...] = (
    "BAR",
    "LINE",
    "SCATTER",
    "HISTOGRAM",
    "COUNT_BAR",
)


class VisualizationCandidate(BaseModel):
    """A single structurally feasible visualization candidate.

    Contains enough information for the next stage (chart generation) to
    build the actual chart without re-examining the data.
    """

    model_config = ConfigDict(frozen=True)

    chart_type: VisualizationType
    x_column: str
    y_column: str | None = None
    signals: list[str] = Field(default_factory=list)


class FeasibilityResult(BaseModel):
    """Complete visualization feasibility analysis for a dataset.

    Produced by :func:`analyze_visualization_feasibility`. Fully deterministic
    for a given (dataframe, classification) pair.
    """

    model_config = ConfigDict(frozen=True)

    dataset_name: str | None = None
    row_count: int
    column_count: int
    candidates: list[VisualizationCandidate] = Field(default_factory=list)
    suppressed_reasons: list[str] = Field(default_factory=list)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def candidates_by_type(self, chart_type: VisualizationType) -> list[VisualizationCandidate]:
        """Return all candidates of a given chart type."""
        return [c for c in self.candidates if c.chart_type == chart_type]

    @property
    def chart_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {t: 0 for t in ALL_VIZ_TYPES}
        for c in self.candidates:
            counts[c.chart_type] += 1
        return counts


# ---------------------------------------------------------------------------
# Stage 3: Visualization Candidate Ranking Schemas
# ---------------------------------------------------------------------------

# Deterministic chart-type priority used as a tiebreaker in sort key.
# Lower number = higher priority when scores are equal.
# Rationale: LINE > BAR > SCATTER > HISTOGRAM > COUNT_BAR reflects the
# information richness typical across most tabular datasets.
CHART_TYPE_PRIORITY: dict[str, int] = {
    "LINE": 0,
    "BAR": 1,
    "SCATTER": 2,
    "HISTOGRAM": 3,
    "COUNT_BAR": 4,
}


class RankedCandidate(BaseModel):
    """A visualization candidate with a deterministic usefulness score.

    Score meaning
    -------------
    ``score`` is a deterministic visualization usefulness score in [0.0, 1.0].
    It is NOT a probability, statistical significance, prediction confidence,
    or model confidence.  It is computed from measurable dataset characteristics
    using fixed, documented weights and is only meaningful for ordering
    candidates relative to each other on the same dataset.

    Fields
    ------
    chart_type, x_column, y_column:
        Directly inherited from the Stage 2 ``VisualizationCandidate``.
    score:
        Deterministic usefulness score in [0.0, 1.0].
    rank:
        1-based position in the final ordered list (1 = best).
    signals:
        Combined feasibility signals from Stage 2 plus ranking signals
        from Stage 3 explaining each score component.
    """

    model_config = ConfigDict(frozen=True)

    chart_type: VisualizationType
    x_column: str
    y_column: str | None = None
    score: float
    rank: int
    signals: list[str] = Field(default_factory=list)


class RankingResult(BaseModel):
    """Complete Stage 3 ranking output for a dataset.

    Produced by :func:`rank_visualization_candidates`. Fully deterministic
    for a given (dataframe, classification, feasibility_result) triple.
    """

    model_config = ConfigDict(frozen=True)

    dataset_name: str | None = None
    row_count: int
    column_count: int
    ranked: list[RankedCandidate] = Field(default_factory=list)
    ranking_notes: list[str] = Field(default_factory=list)

    @property
    def top(self) -> RankedCandidate | None:
        """Return the highest-ranked candidate, or None if empty."""
        return self.ranked[0] if self.ranked else None

    @property
    def ranked_count(self) -> int:
        return len(self.ranked)

    def ranked_by_type(self, chart_type: VisualizationType) -> list[RankedCandidate]:
        """Return ranked candidates filtered to a specific chart type."""
        return [c for c in self.ranked if c.chart_type == chart_type]


# ---------------------------------------------------------------------------
# Stage 4: Chart Data Generation Schemas
# ---------------------------------------------------------------------------

# Aggregation functions used by BAR and LINE chart data.
AggregationMethod = Literal["sum", "mean", "count", "median", "min", "max", None]

# Hard limits used by Stage 4 data_builder — exported so tests can import them.
# All limits are documented with rationale in data_builder.py.
STAGE4_MAX_CATEGORY_POINTS: int = 50   # BAR / COUNT_BAR: max distinct category bars
STAGE4_MAX_LINE_POINTS: int = 500      # LINE: max chronological data points
STAGE4_MAX_SCATTER_POINTS: int = 2000  # SCATTER: max row-wise pairs
STAGE4_MAX_HISTOGRAM_BINS: int = 50    # HISTOGRAM: max bins


class ChartPoint(BaseModel):
    """A single (x, y) data point for BAR, LINE, or SCATTER charts."""

    model_config = ConfigDict(frozen=True)

    x: float | int | str
    y: float | int


class HistogramBin(BaseModel):
    """A single histogram bin with explicit boundaries and observation count."""

    model_config = ConfigDict(frozen=True)

    bin_start: float
    bin_end: float
    count: int


class ChartData(BaseModel):
    """The complete, JSON-serializable chart data payload produced by Stage 4.

    This is the direct input to any chart renderer.  It contains no pandas
    or numpy objects — all values are native Python scalars.

    Fields
    ------
    chart_type:
        The chart type, matches the Stage 3 ``RankedCandidate``.
    x_column:
        The column name used for the x-axis.
    y_column:
        The column name used for the y-axis (None for HISTOGRAM and COUNT_BAR).
    aggregation:
        How the measure was aggregated (``"sum"``, ``"mean"``, ``"count"``, or None).
    points:
        For BAR, LINE, SCATTER, COUNT_BAR: list of ``ChartPoint`` objects.
        For HISTOGRAM: empty list (see ``bins``).
    bins:
        For HISTOGRAM: list of ``HistogramBin`` objects.
        For other chart types: empty list.
    signals:
        Deterministic processing signals explaining what was done.
    usable_row_count:
        Number of rows used after filtering missing / non-finite values.
    sampled:
        True if the payload was deterministically reduced from a larger dataset.
    has_data:
        False when no usable chart data could be produced (e.g. all-missing).
    """

    model_config = ConfigDict(frozen=True)

    chart_type: VisualizationType
    x_column: str
    y_column: str | None = None
    aggregation: AggregationMethod = None
    points: list[ChartPoint] = Field(default_factory=list)
    bins: list[HistogramBin] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    usable_row_count: int = 0
    sampled: bool = False
    has_data: bool = True


# ---------------------------------------------------------------------------
# Stage 5: Visualization Orchestrator Schemas
# ---------------------------------------------------------------------------

VisualizationStatus = Literal["READY", "NO_MEANINGFUL_VISUALIZATION"]


class VisualizationItem(BaseModel):
    """A single automatic visualization result with ranking and chart data."""

    model_config = ConfigDict(frozen=True)

    rank: int
    chart_type: VisualizationType
    x_column: str
    y_column: str | None = None
    score: float
    chart_data: ChartData
    signals: list[str] = Field(default_factory=list)


class VisualizationResult(BaseModel):
    """Complete automatic visualization result produced by Stage 5 orchestrator."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str | None = None
    row_count: int
    column_count: int
    status: VisualizationStatus
    visualizations: list[VisualizationItem] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)

    @property
    def visualization_count(self) -> int:
        return len(self.visualizations)

    @property
    def top(self) -> VisualizationItem | None:
        return self.visualizations[0] if self.visualizations else None


# ---------------------------------------------------------------------------
# Stage 7: Visualization Request Intelligence Schemas
# ---------------------------------------------------------------------------

VisualizationIntent = Literal["DISTRIBUTION", "COMPARISON", "TREND", "RELATIONSHIP"]
ALL_INTENTS: tuple[VisualizationIntent, ...] = (
    "DISTRIBUTION",
    "COMPARISON",
    "TREND",
    "RELATIONSHIP",
)

RequestAggregation = Literal["COUNT", "SUM", "MEAN", "MEDIAN", "MIN", "MAX"]
ALL_REQUEST_AGGREGATIONS: tuple[RequestAggregation, ...] = (
    "COUNT",
    "SUM",
    "MEAN",
    "MEDIAN",
    "MIN",
    "MAX",
)

VisualizationRequestStatus = Literal["VALID", "INVALID", "NEEDS_CLARIFICATION"]


class VisualizationRequest(BaseModel):
    """Parsed visualization request from natural language text.

    Produced by :func:`parse_visualization_request`.
    """

    model_config = ConfigDict(frozen=True)

    raw_request: str
    requested_chart_type: VisualizationType | None = None
    is_explicit_chart_type: bool = False
    x_column: str | None = None
    y_column: str | None = None
    aggregation: RequestAggregation | None = None
    intent: VisualizationIntent | None = None
    requested_columns: list[str] = Field(default_factory=list)
    parser_signals: list[str] = Field(default_factory=list)


class ValidatedVisualizationRequest(BaseModel):
    """Fully validated visualization request ready for execution.

    Produced by :func:`validate_visualization_request`.
    """

    model_config = ConfigDict(frozen=True)

    status: VisualizationRequestStatus
    chart_type: VisualizationType | None = None
    x_column: str | None = None
    y_column: str | None = None
    aggregation: RequestAggregation | None = None
    intent: VisualizationIntent | None = None
    requested_columns: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    clarification_options: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert validated request to a deterministic, JSON-serializable dictionary."""
        return {
            "status": str(self.status),
            "chart_type": str(self.chart_type) if self.chart_type is not None else None,
            "x_column": str(self.x_column) if self.x_column is not None else None,
            "y_column": str(self.y_column) if self.y_column is not None else None,
            "aggregation": str(self.aggregation) if self.aggregation is not None else None,
            "intent": str(self.intent) if self.intent is not None else None,
            "requested_columns": [str(c) for c in self.requested_columns],
            "signals": [str(s) for s in self.signals],
            "errors": [str(e) for e in self.errors],
            "clarification_options": [str(opt) for opt in self.clarification_options],
        }


