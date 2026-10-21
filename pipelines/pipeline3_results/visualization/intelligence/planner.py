"""One downstream path from a VerifiedResult to plan and chart specification."""

from __future__ import annotations

import hashlib
import json

from pipelines.pipeline3_results.execution.schemas import VerifiedResult
from pipelines.pipeline3_results.grounding.schemas import DatasetKnowledge
from pipelines.pipeline3_results.insights.landing.opportunities import display_label
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan
from .schemas import (
    ChartData, ChartFamily, ChartSeries, DashboardVisualization, FieldEncoding,
    RendererNeutralChartSpec, ScoreComponent, VisualizationPlan, VisualizationPurpose,
)
from .suitability import assess_suitability, default_family, purpose_for

_MAX_INLINE_ROWS = 500


def _identity(*values: str) -> str:
    raw = json.dumps(values, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _label(field: str, knowledge: DatasetKnowledge) -> str:
    column = knowledge.get_column(field)
    return display_label(column) if column is not None else field.replace("_", " ").title()


def _format(field: str, knowledge: DatasetKnowledge) -> str:
    column = knowledge.get_column(field)
    semantic = (column.semantic_type if column else "") or ""
    value_function = (column.value_function if column else "") or ""
    if semantic.casefold() == "percentage":
        return "PERCENT"
    if semantic.casefold() == "currency" or value_function.casefold() == "currency":
        return "CURRENCY"
    if semantic.casefold() == "date":
        return "DATE"
    if semantic.casefold() in {"datetime", "time"}:
        return "DATETIME"
    dtype = (column.physical_dtype if column else "") or ""
    return "INTEGER" if "int" in dtype.casefold() else "NUMBER"


def _encodings(plan: AnalyticalPlan, result: VerifiedResult, knowledge: DatasetKnowledge) -> tuple[FieldEncoding | None, FieldEncoding | None]:
    if result.result_type == "SERIES":
        x = plan.temporal[0].column
        y = "value"
        return (
            FieldEncoding(channel="X", field=x, semantic_label=_label(x, knowledge), data_type="temporal", role="TEMPORAL", format_hint=_format(x, knowledge)),
            FieldEncoding(channel="Y", field=y, semantic_label=_label(plan.measures[0].column or "value", knowledge), data_type="numeric", role="MEASURE", source_field=plan.measures[0].column, format_hint=_format(plan.measures[0].column or "value", knowledge)),
        )
    if result.result_type == "RELATIONSHIP":
        left, right = plan.measures[0].column or "x", plan.measures[1].column or "y"
        return (
            FieldEncoding(channel="X", field=left, semantic_label=_label(left, knowledge), data_type="numeric", role="MEASURE", source_field=left, format_hint=_format(left, knowledge)),
            FieldEncoding(channel="Y", field=right, semantic_label=_label(right, knowledge), data_type="numeric", role="MEASURE", source_field=right, format_hint=_format(right, knowledge)),
        )
    if result.result_type == "DISTRIBUTION":
        field = plan.measures[0].column or "value"
        if result.execution_metadata.details.get("distribution_kind") == "numeric_bins":
            return (
                FieldEncoding(channel="X", field="bin_start", semantic_label=_label(field, knowledge), data_type="numeric_bin", role="BIN", format_hint=_format(field, knowledge)),
                FieldEncoding(channel="Y", field="count", semantic_label="Count", data_type="integer", role="COUNT", format_hint="INTEGER"),
            )
        return (
            FieldEncoding(channel="X", field=field, semantic_label=_label(field, knowledge), data_type="categorical", role="DIMENSION", format_hint=_format(field, knowledge)),
            FieldEncoding(channel="Y", field="count", semantic_label="Count", data_type="integer", role="COUNT", format_hint="INTEGER"),
        )
    dimension = plan.dimensions[0].column if plan.dimensions else "category"
    measure = plan.measures[0].column or "value"
    return (
        FieldEncoding(channel="X", field=dimension, semantic_label=_label(dimension, knowledge), data_type="categorical", role="DIMENSION", format_hint=_format(dimension, knowledge)),
        FieldEncoding(channel="Y", field="value", semantic_label=_label(measure, knowledge) if measure != "value" else "Count", data_type="numeric", role="MEASURE", source_field=(measure if measure != "value" else None), format_hint=_format(measure, knowledge) if measure != "value" else "INTEGER"),
    )


def _title(plan: AnalyticalPlan, result: VerifiedResult, knowledge: DatasetKnowledge) -> str:
    purpose = purpose_for(plan, result)
    measure = plan.measures[0].column if plan.measures else None
    measure_label = _label(measure, knowledge) if measure else "Records"
    if plan.measures and plan.measures[0].kind == "COUNT_ROWS":
        measure_label = "Record Count"
    if purpose.value == "TREND":
        return f"{measure_label} Trend"
    if purpose.value == "DISTRIBUTION":
        return f"{measure_label} Distribution"
    if purpose.value == "RELATIONSHIP":
        second = _label(plan.measures[1].column or "value", knowledge)
        return f"{measure_label} vs {second}"
    if plan.dimensions:
        return f"{measure_label} by {_label(plan.dimensions[0].column, knowledge)}"
    return measure_label


def build_visualization(
    plan: AnalyticalPlan,
    result: VerifiedResult,
    knowledge: DatasetKnowledge,
    *,
    source: str,
    requested_family: ChartFamily | None = None,
    score: float | None = None,
    score_components: list[ScoreComponent] | None = None,
) -> DashboardVisualization | None:
    if result.status != "READY" or result.result_type in {"SCALAR", "STATISTIC", "EXPLORATORY_PLACEHOLDER"}:
        return None
    purpose = purpose_for(plan, result)
    family = requested_family or default_family(purpose)
    if family in {ChartFamily.PIE, ChartFamily.DOUGHNUT}:
        purpose = VisualizationPurpose.PART_TO_WHOLE
    suitability = assess_suitability(plan, result, knowledge, family)
    if suitability.status in {"INVALID", "NOT_RECOMMENDED"}:
        return None
    x_encoding, y_encoding = _encodings(plan, result, knowledge)
    visual_id = f"visualization_{_identity(plan.plan_id, result.result_id, family.value)}"
    warnings = [suitability.message] if suitability.status == "SUITABLE_WITH_WARNING" else []
    vplan = VisualizationPlan(
        visualization_id=visual_id, source=source, analytical_plan_id=plan.plan_id,
        verified_result_id=result.result_id, chart_family=family, purpose=purpose,
        x_encoding=x_encoding, y_encoding=y_encoding,
        aggregation_context=(plan.measures[0].operation if plan.measures else None),
        sort_context=(plan.sort[0].direction if plan.sort else None),
        temporal_context=(plan.temporal[0].time_granularity if plan.temporal else None),
        suitability=suitability,
        why_selected=[*suitability.reason_codes, "VERIFIED_RESULT_REUSED"], warnings=warnings,
    )
    rows = result.rows[:_MAX_INLINE_ROWS]
    truncated = len(result.rows) > _MAX_INLINE_ROWS
    if truncated:
        warnings.append("Chart data is presentation-bounded; VerifiedResult remains complete.")
    encodings = [item for item in (x_encoding, y_encoding) if item is not None]
    orientation = "HORIZONTAL" if family == ChartFamily.BAR and len(result.rows) > 8 else ("RADIAL" if family in {ChartFamily.PIE, ChartFamily.DOUGHNUT} else "CARTESIAN")
    palette = "CATEGORICAL" if family in {ChartFamily.PIE, ChartFamily.DOUGHNUT} else "SINGLE_SERIES"
    interactions = ["HOVER", "EXPORT"]
    if len(result.rows) > 20:
        interactions = ["HOVER", "ZOOM", "PAN", "DATA_VIEW", "EXPORT"]
    data = ChartData(
        columns=[field.name for field in result.schema_.fields], rows=rows,
        coverage=result.coverage.model_dump(mode="json"), truncated=truncated,
        display_limit=(_MAX_INLINE_ROWS if truncated else None),
        source_result_id=result.result_id,
    )
    title = _title(plan, result, knowledge)
    series = []
    if y_encoding is not None:
        series = [ChartSeries(id="series_1", label=y_encoding.semantic_label, measure_field=y_encoding.field)]
    spec = RendererNeutralChartSpec(
        chart_id=f"chart_{_identity(visual_id)}", family=family, title=title,
        subtitle=None, encodings=encodings, series=series,
        orientation=orientation, stacking="NONE",
        legend="VISIBLE" if family in {ChartFamily.PIE, ChartFamily.DOUGHNUT} else "AUTO",
        tooltip=True, labels="PERCENT" if family in {ChartFamily.PIE, ChartFamily.DOUGHNUT} else "AUTO",
        number_format=(y_encoding.format_hint if y_encoding else "NUMBER"),
        palette_role=palette, interaction_hints=interactions, data_ref=data,
        warnings=warnings,
    )
    return DashboardVisualization(
        id=visual_id, source=source, title=title, why_selected=vplan.why_selected,
        analytical_plan_ref=plan.plan_id, verified_result_ref=result.result_id,
        score=score, score_components=score_components or [], plan=vplan, chart_spec=spec,
    )


__all__ = ["build_visualization"]
