"""Chart-aware deterministic fact extraction from a verified analytical result."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Any, Iterable

from pipelines.pipeline3_results.execution.schemas import VerifiedResult
from pipelines.pipeline3_results.visualization.intelligence.schemas import (
    ChartFamily, RendererNeutralChartSpec, VisualizationPlan,
)
from .schemas import FactType, XAIFact


_ADVANCED = {
    ChartFamily.TREEMAP, ChartFamily.SUNBURST, ChartFamily.FUNNEL,
    ChartFamily.RADAR, ChartFamily.GAUGE, ChartFamily.SANKEY,
    ChartFamily.COMBINATION,
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float):
        return round(value, 12)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _fact(
    fact_type: FactType,
    importance: float,
    values: dict[str, Any],
    result: VerifiedResult,
    spec: RendererNeutralChartSpec,
    *,
    caveats: Iterable[str] = (),
) -> XAIFact:
    safe_values = _safe(values)
    digest = hashlib.sha256(json.dumps(
        [fact_type, safe_values, result.result_id, spec.chart_id],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()[:16]
    return XAIFact(
        fact_id=f"xai_fact_{digest}", fact_type=fact_type,
        importance=importance, values=safe_values,
        evidence_refs=[result.result_id, spec.chart_id], caveats=list(caveats),
    )


def _labels(plan: VisualizationPlan) -> tuple[str, str]:
    x = plan.x_encoding.semantic_label if plan.x_encoding else "Category"
    y = plan.y_encoding.semantic_label if plan.y_encoding else "Value"
    return x, y


def _coverage(
    result: VerifiedResult, spec: RendererNeutralChartSpec,
) -> tuple[list[XAIFact], list[str]]:
    coverage = result.coverage
    facts: list[XAIFact] = []
    limitations: list[str] = []
    excluded = coverage.excluded_null_rows + coverage.excluded_invalid_rows
    if coverage.source_rows and excluded:
        pct = excluded / coverage.source_rows * 100.0
        if pct >= 1.0:
            note = f"{pct:.1f}% of source rows were excluded because values were missing or invalid."
            limitations.append(note)
            facts.append(_fact("COVERAGE", .62, {
                "source_rows": coverage.source_rows,
                "usable_rows": coverage.usable_rows,
                "excluded_rows": excluded,
                "excluded_percent": round(pct, 1),
            }, result, spec, caveats=[note]))
    if spec.data_ref.truncated:
        limit = spec.data_ref.display_limit or len(spec.data_ref.rows)
        limitations.append(f"Only the first {limit:,} verified result rows are displayed in the chart.")
    return facts, limitations


def _category_values(
    result: VerifiedResult, plan: VisualizationPlan,
) -> list[tuple[Any, float]]:
    x_field = plan.x_encoding.field if plan.x_encoding else "category"
    y_field = plan.y_encoding.field if plan.y_encoding else "value"
    return [
        (row.get(x_field), value)
        for row in result.rows
        if (value := _number(row.get(y_field))) is not None
    ]


def _bars(
    result: VerifiedResult, plan: VisualizationPlan,
    spec: RendererNeutralChartSpec,
) -> tuple[list[XAIFact], list[str]]:
    category_label, value_label = _labels(plan)
    values = _category_values(result, plan)
    if not values:
        return [], ["No verified numeric category values were available to explain."]
    ordered = sorted(values, key=lambda item: (-item[1], str(item[0])))
    maximum, minimum = ordered[0][1], ordered[-1][1]
    leaders = [category for category, value in ordered if value == maximum]
    low = [category for category, value in ordered if value == minimum]
    facts = [
        _fact("MAXIMUM", 1.0, {"category": leaders[0], "categories": leaders, "value": maximum, "category_label": category_label, "value_label": value_label}, result, spec),
        _fact("MINIMUM", .72, {"category": low[0], "categories": low, "value": minimum, "category_label": category_label, "value_label": value_label}, result, spec),
        _fact("RANKING", .86, {"order": [category for category, _ in ordered], "values": [value for _, value in ordered], "category_count": len(ordered), "category_label": category_label, "value_label": value_label}, result, spec),
        _fact("ABSOLUTE_CHANGE", .78, {"from_category": low[0], "to_category": leaders[0], "difference": maximum - minimum, "value_label": value_label}, result, spec),
    ]
    if len(leaders) > 1 or len(low) > 1:
        facts.append(_fact("TIE", .82, {"highest": leaders, "lowest": low, "value": maximum if len(leaders) > 1 else minimum, "category_label": category_label}, result, spec))
    if minimum != 0:
        facts.append(_fact("PERCENT_CHANGE", .7, {"from_category": low[0], "to_category": leaders[0], "percent": (maximum - minimum) / abs(minimum) * 100.0, "denominator": minimum}, result, spec))
    total = sum(value for _, value in values)
    if total > 0 and all(value >= 0 for _, value in values):
        top_n = min(3, len(ordered))
        facts.extend([
            _fact("TOP_SHARE", .84, {"category": leaders[0], "share_percent": maximum / total * 100.0, "total": total, "value_label": value_label}, result, spec),
            _fact("CONCENTRATION", .68, {"top_n": top_n, "share_percent": sum(value for _, value in ordered[:top_n]) / total * 100.0, "category_count": len(ordered)}, result, spec),
        ])
    if spec.family in {ChartFamily.GROUPED_BAR, ChartFamily.STACKED_BAR} and len(spec.series) > 1:
        series_values: list[tuple[Any, str, float]] = []
        x_field = plan.x_encoding.field if plan.x_encoding else "category"
        for row in result.rows:
            for series in spec.series:
                value = _number(row.get(series.measure_field))
                if value is not None:
                    series_values.append((row.get(x_field), series.label, value))
        if series_values:
            highest = max(series_values, key=lambda item: item[2])
            facts.append(_fact("MAXIMUM", .96, {"category": highest[0], "series": highest[1], "value": highest[2], "value_label": value_label}, result, spec))
            by_category: dict[str, list[tuple[str, float]]] = {}
            for category, series, value in series_values:
                by_category.setdefault(str(category), []).append((series, value))
            gaps = [(category, max(items, key=lambda i: i[1]), min(items, key=lambda i: i[1])) for category, items in by_category.items() if len(items) > 1]
            if gaps:
                category, high, low_item = max(gaps, key=lambda item: item[1][1] - item[2][1])
                facts.append(_fact("ABSOLUTE_CHANGE", .8, {"category": category, "higher_series": high[0], "lower_series": low_item[0], "difference": high[1] - low_item[1]}, result, spec))
            if spec.family == ChartFamily.STACKED_BAR and all(item[2] >= 0 for item in series_values):
                totals = {
                    category: sum(value for _, value in items)
                    for category, items in by_category.items()
                }
                component_totals: dict[str, float] = {}
                largest_components: list[dict[str, Any]] = []
                for category, items in by_category.items():
                    for series, value in items:
                        component_totals[series] = component_totals.get(series, 0.0) + value
                    largest = max(items, key=lambda item: item[1])
                    largest_components.append({"category": category, "series": largest[0], "value": largest[1]})
                total = sum(component_totals.values())
                facts.append(_fact("RANKING", .88, {
                    "group_totals": totals,
                    "order": sorted(totals, key=lambda name: (-totals[name], name)),
                    "largest_components": largest_components,
                    "value_label": value_label,
                }, result, spec))
                if total > 0:
                    leading_component = max(component_totals, key=component_totals.get)  # type: ignore[arg-type]
                    facts.append(_fact("TOP_SHARE", .8, {
                        "series": leading_component,
                        "share_percent": component_totals[leading_component] / total * 100.0,
                        "component_totals": component_totals,
                    }, result, spec))
    return facts, []


def _line(
    result: VerifiedResult, plan: VisualizationPlan,
    spec: RendererNeutralChartSpec,
) -> tuple[list[XAIFact], list[str]]:
    period_label, value_label = _labels(plan)
    values = _category_values(result, plan)
    if not values:
        return [], ["No verified ordered numeric values were available to explain."]
    first, last = values[0], values[-1]
    change = last[1] - first[1]
    direction = "flat" if change == 0 else ("increased" if change > 0 else "decreased")
    high = max(values, key=lambda item: item[1])
    low = min(values, key=lambda item: item[1])
    facts = [
        _fact("ABSOLUTE_CHANGE", .98, {"first_period": first[0], "first_value": first[1], "last_period": last[0], "last_value": last[1], "change": change, "period_count": len(values), "period_label": period_label, "value_label": value_label}, result, spec),
        _fact("TREND_DIRECTION", .9, {"direction": direction, "period_count": len(values), "value_label": value_label}, result, spec),
        _fact("MAXIMUM", .82, {"period": high[0], "value": high[1], "value_label": value_label}, result, spec),
        _fact("MINIMUM", .72, {"period": low[0], "value": low[1], "value_label": value_label}, result, spec),
    ]
    limitations: list[str] = []
    if first[1] != 0:
        facts.append(_fact("PERCENT_CHANGE", .84, {"first_period": first[0], "last_period": last[0], "percent": change / abs(first[1]) * 100.0, "denominator": first[1]}, result, spec))
    else:
        limitations.append("Percentage change is not reported because the first value is zero.")
    adjacent = [(values[i][0], values[i + 1][0], values[i + 1][1] - values[i][1]) for i in range(len(values) - 1)]
    if adjacent:
        inc = max(adjacent, key=lambda item: item[2])
        dec = min(adjacent, key=lambda item: item[2])
        facts.append(_fact("LARGEST_INCREASE", .76, {"from_period": inc[0], "to_period": inc[1], "change": inc[2]}, result, spec))
        facts.append(_fact("LARGEST_DECREASE", .74, {"from_period": dec[0], "to_period": dec[1], "change": dec[2]}, result, spec))
        if len(adjacent) > 1:
            facts.append(_fact("RANGE", .55, {"adjacent_change_standard_deviation": statistics.pstdev(item[2] for item in adjacent), "metric": "population standard deviation of adjacent changes"}, result, spec))
    if plan.temporal_context is None:
        limitations.append("The sequence is treated as ordered by the verified result; no time granularity was supplied.")
    return facts, limitations


def _pie(result: VerifiedResult, plan: VisualizationPlan, spec: RendererNeutralChartSpec) -> tuple[list[XAIFact], list[str]]:
    values = _category_values(result, plan)
    total = sum(value for _, value in values)
    if not values or total <= 0 or any(value < 0 for _, value in values):
        return [], ["Part-to-whole facts require non-negative values with a positive displayed total."]
    facts, limitations = _bars(result, plan, spec)
    return facts, limitations


def _scatter(result: VerifiedResult, plan: VisualizationPlan, spec: RendererNeutralChartSpec) -> tuple[list[XAIFact], list[str]]:
    x_field = plan.x_encoding.field if plan.x_encoding else "x"
    y_field = plan.y_encoding.field if plan.y_encoding else "y"
    x_label, y_label = _labels(plan)
    pairs = [(x, y) for row in result.rows if (x := _number(row.get(x_field))) is not None and (y := _number(row.get(y_field))) is not None]
    limitations = ["Correlation does not establish causation.", f"The chart is based on {len(pairs):,} valid pairs."]
    facts = [_fact("PAIR_COUNT", .82, {"pair_count": len(pairs), "x_label": x_label, "y_label": y_label}, result, spec)]
    if len(pairs) < 2:
        limitations.append("Pearson correlation could not be computed because fewer than two valid pairs were available.")
        return facts, limitations
    xs, ys = zip(*pairs)
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    if denominator == 0:
        limitations.append("Pearson correlation could not be computed because one variable is constant.")
        return facts, limitations
    correlation = numerator / denominator
    magnitude = abs(correlation)
    strength = "very weak" if magnitude < .2 else "weak" if magnitude < .4 else "moderate" if magnitude < .6 else "strong" if magnitude < .8 else "very strong"
    direction = "positive" if correlation > 0 else ("negative" if correlation < 0 else "none")
    facts.append(_fact("CORRELATION", 1.0, {"coefficient": correlation, "direction": direction, "strength": strength, "pair_count": len(pairs), "method": "Pearson", "x_label": x_label, "y_label": y_label}, result, spec, caveats=["Correlation does not establish causation."]))
    return facts, limitations


def _histogram(result: VerifiedResult, plan: VisualizationPlan, spec: RendererNeutralChartSpec) -> tuple[list[XAIFact], list[str]]:
    bins: list[tuple[float, float, float]] = []
    for row in result.rows:
        start, end, count = _number(row.get("bin_start")), _number(row.get("bin_end")), _number(row.get("count"))
        if start is not None and end is not None and count is not None:
            bins.append((start, end, count))
    if not bins:
        return [], ["No verified histogram bins were available to explain."]
    dominant = max(bins, key=lambda item: item[2])
    lower, upper = min(item[0] for item in bins), max(item[1] for item in bins)
    gaps = sum(1 for _, _, count in bins if count == 0)
    label = plan.x_encoding.semantic_label if plan.x_encoding else "Value"
    facts = [
        _fact("DOMINANT_BIN", 1.0, {"bin_start": dominant[0], "bin_end": dominant[1], "count": dominant[2], "value_label": label}, result, spec),
        _fact("RANGE", .8, {"minimum": lower, "maximum": upper, "spread": upper - lower, "bin_count": len(bins), "empty_bin_count": gaps, "value_label": label}, result, spec),
    ]
    total = sum(item[2] for item in bins)
    if total > 0:
        center = sum(((start + end) / 2) * count for start, end, count in bins) / total
        range_values = dict(facts[1].values)
        range_values.update({"center_estimate": center, "center_method": "frequency-weighted bin midpoint"})
        facts[1] = _fact("RANGE", .8, range_values, result, spec, caveats=["The center is a bin-based estimate, not an exact median."])
    return facts, ["Distribution shape is described from verified bins; normality is not inferred."]


def _box(result: VerifiedResult, plan: VisualizationPlan, spec: RendererNeutralChartSpec) -> tuple[list[XAIFact], list[str]]:
    facts: list[XAIFact] = []
    x_field = plan.x_encoding.field if plan.x_encoding else "group"
    for row in result.rows:
        median, q1, q3 = _number(row.get("median")), _number(row.get("q1")), _number(row.get("q3"))
        if median is None or q1 is None or q3 is None:
            continue
        group = row.get(x_field)
        facts.extend([
            _fact("MEDIAN", .92, {"group": group, "median": median}, result, spec),
            _fact("IQR", .84, {"group": group, "q1": q1, "q3": q3, "iqr": q3 - q1, "whisker_low": _number(row.get("whisker_low", row.get("low"))), "whisker_high": _number(row.get("whisker_high", row.get("high")))}, result, spec),
        ])
        count = _number(row.get("outlier_count"))
        if count is not None:
            facts.append(_fact("OUTLIER_COUNT", .75, {"group": group, "count": int(count), "rule": "box-plot rule"}, result, spec))
    return (facts, [] if facts else ["The verified result does not contain a supported box-summary shape."])


def _heatmap(result: VerifiedResult, plan: VisualizationPlan, spec: RendererNeutralChartSpec) -> tuple[list[XAIFact], list[str]]:
    fields = [item.field for item in spec.encodings]
    if len(fields) < 3:
        return [], ["The verified result does not contain a supported heatmap cell shape."]
    x_field, y_field, value_field = fields[0], fields[1], fields[-1]
    cells = [(row.get(x_field), row.get(y_field), value) for row in result.rows if (value := _number(row.get(value_field))) is not None]
    if not cells:
        return [], ["No verified numeric heatmap cells were available to explain."]
    high, low = max(cells, key=lambda item: item[2]), min(cells, key=lambda item: item[2])
    facts = [
        _fact("HIGH_VALUE_CELL", 1.0, {"x": high[0], "y": high[1], "value": high[2]}, result, spec),
        _fact("LOW_VALUE_CELL", .82, {"x": low[0], "y": low[1], "value": low[2]}, result, spec),
    ]
    total = sum(item[2] for item in cells)
    if total > 0 and all(item[2] >= 0 for item in cells):
        row_totals: dict[str, float] = {}
        column_totals: dict[str, float] = {}
        for x, y, value in cells:
            row_totals[str(y)] = row_totals.get(str(y), 0.0) + value
            column_totals[str(x)] = column_totals.get(str(x), 0.0) + value
        facts.append(_fact("CONCENTRATION", .66, {
            "leading_row": max(row_totals, key=row_totals.get),  # type: ignore[arg-type]
            "leading_column": max(column_totals, key=column_totals.get),  # type: ignore[arg-type]
            "row_totals": row_totals, "column_totals": column_totals,
        }, result, spec))
    return facts, []


def analyze_chart(
    result: VerifiedResult, plan: VisualizationPlan,
    spec: RendererNeutralChartSpec,
) -> tuple[list[XAIFact], list[str]]:
    """Return deterministic facts and useful limitations for one chart."""
    coverage_facts, coverage_limits = _coverage(result, spec)
    if spec.family in _ADVANCED:
        return coverage_facts, [*coverage_limits, f"Explainable analytics is not yet supported for {spec.family.value.replace('_', ' ').lower()} charts."]
    if spec.family in {ChartFamily.BAR, ChartFamily.GROUPED_BAR, ChartFamily.STACKED_BAR}:
        facts, limitations = _bars(result, plan, spec)
    elif spec.family in {ChartFamily.LINE, ChartFamily.AREA}:
        facts, limitations = _line(result, plan, spec)
    elif spec.family in {ChartFamily.PIE, ChartFamily.DOUGHNUT}:
        facts, limitations = _pie(result, plan, spec)
    elif spec.family == ChartFamily.SCATTER:
        facts, limitations = _scatter(result, plan, spec)
    elif spec.family == ChartFamily.HISTOGRAM:
        facts, limitations = _histogram(result, plan, spec)
    elif spec.family == ChartFamily.BOX_PLOT:
        facts, limitations = _box(result, plan, spec)
    elif spec.family == ChartFamily.HEATMAP:
        facts, limitations = _heatmap(result, plan, spec)
    else:
        facts, limitations = [], ["Explainable analytics is unavailable for this chart family."]
    for warning in [*result.warnings, *plan.warnings, *spec.warnings]:
        if warning and warning not in limitations:
            limitations.append(warning)
    return [*facts, *coverage_facts], [*limitations, *coverage_limits]


def select_prose_facts(facts: list[XAIFact], *, limit: int = 4) -> list[XAIFact]:
    """Choose high-value, non-redundant facts while retaining all facts in XAIResult."""
    selected: list[XAIFact] = []
    seen: set[str] = set()
    available = {item.fact_type for item in facts}
    for fact in sorted(facts, key=lambda item: (-item.importance, item.fact_id)):
        if fact.fact_type in {"COVERAGE", "LIMITATION"} or fact.fact_type in seen:
            continue
        if fact.fact_type == "PAIR_COUNT" and "CORRELATION" in available:
            continue
        selected.append(fact)
        seen.add(fact.fact_type)
        if len(selected) == limit:
            break
    return selected


__all__ = ["analyze_chart", "select_prose_facts"]
