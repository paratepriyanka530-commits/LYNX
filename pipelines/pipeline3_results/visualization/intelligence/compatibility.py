"""Compatibility shaping for the existing public visualization response."""

from __future__ import annotations

from .schemas import DashboardVisualization


def to_legacy_item(visual: DashboardVisualization, *, rank: int) -> dict:
    spec = visual.chart_spec
    family = spec.family.value
    chart_type = "PIE" if family == "DOUGHNUT" else family
    if chart_type not in {"BAR", "LINE", "SCATTER", "HISTOGRAM", "PIE"}:
        chart_type = "BAR"
    x = visual.plan.x_encoding.field if visual.plan.x_encoding else ""
    y_data = visual.plan.y_encoding.field if visual.plan.y_encoding else None
    y = (
        visual.plan.y_encoding.source_field
        if visual.plan.y_encoding and y_data == "value"
        else y_data
    )
    points: list[dict] = []
    bins: list[dict] = []
    if chart_type == "HISTOGRAM":
        bins = [
            {"bin_start": row["bin_start"], "bin_end": row["bin_end"], "count": row["count"]}
            for row in spec.data_ref.rows
        ]
    else:
        points = [{"x": row.get(x), "y": row.get(y_data)} for row in spec.data_ref.rows]
    signals = [
        f"analytical_plan:{visual.analytical_plan_ref}",
        f"verified_result:{visual.verified_result_ref}",
        "visualization_intelligence:phase7",
        "visualization_compatibility:no_recomputation",
    ]
    return {
        "rank": rank,
        "chart_type": chart_type,
        "presentation_variant": "DOUGHNUT" if family == "DOUGHNUT" else None,
        "x_column": x,
        "y_column": y,
        "score": visual.score,
        "source": visual.source,
        "title": visual.title,
        "why_selected": list(visual.why_selected),
        "score_components": [item.model_dump(mode="json") for item in visual.score_components],
        "visualization_plan": visual.plan.model_dump(mode="json"),
        "chart_spec": visual.chart_spec.model_dump(mode="json"),
        "chart_data": {
            "chart_type": chart_type,
            "presentation_variant": "DOUGHNUT" if family == "DOUGHNUT" else None,
            "x_column": x,
            "y_column": y,
            "aggregation": visual.plan.aggregation_context.casefold() if visual.plan.aggregation_context else None,
            "points": points,
            "bins": bins,
            "signals": signals,
            "usable_row_count": spec.data_ref.coverage["usable_rows"],
            "sampled": spec.data_ref.truncated,
            "has_data": bool(points or bins),
        },
        "signals": signals,
    }


__all__ = ["to_legacy_item"]
