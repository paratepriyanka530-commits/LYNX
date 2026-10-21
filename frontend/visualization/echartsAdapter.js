/** Renderer boundary: RendererNeutralChartSpec -> Apache ECharts option. */
(function (root, factory) {
  const formatter = typeof module === "object" && module.exports
    ? require("./chartFormatter.js") : root.LYNXChartFormatter;
  const theme = typeof module === "object" && module.exports
    ? require("./chartTheme.js") : root.LYNXChartTheme;
  const api = factory(formatter, theme, root);
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.LYNXEChartsAdapter = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function (formatter, theme, root) {
  "use strict";

  const SUPPORTED = new Set(["BAR", "GROUPED_BAR", "STACKED_BAR", "LINE", "AREA", "PIE", "DOUGHNUT", "SCATTER", "HISTOGRAM", "BOX_PLOT", "HEATMAP"]);
  const instances = new WeakMap();
  const observers = new WeakMap();
  const escapeHtml = (value) => String(value == null ? "" : value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");

  function normalizeRendererInput(input) {
    const spec = input && input.chart_spec ? input.chart_spec : input;
    if (!spec || typeof spec !== "object") return { status: "INVALID", message: "No chart specification was provided." };
    const family = String(spec.family || "").toUpperCase();
    if (!SUPPORTED.has(family)) return { status: "UNSUPPORTED", family, message: `${family || "This chart family"} is not supported by the current renderer.` };
    const data = spec.data_ref || {};
    const rows = Array.isArray(data.rows) ? data.rows : [];
    if (!rows.length) return { status: "EMPTY", family, spec, rows, message: "No verified chart data was returned." };
    const encodings = Array.isArray(spec.encodings) ? spec.encodings : [];
    const byChannel = (channel) => encodings.find((item) => String(item.channel).toUpperCase() === channel) || null;
    return { status: "READY", family, spec, rows, x: byChannel("X") || byChannel("CATEGORY"), y: byChannel("Y") || byChannel("VALUE"), color: byChannel("COLOR") || byChannel("SERIES") };
  }

  const valueAt = (row, encoding) => encoding ? row[encoding.field] : undefined;
  const format = (value, encoding, compact = false) => formatter.formatValue(value, encoding && encoding.format_hint, { compact, currency: encoding && encoding.currency, unit: encoding && encoding.unit });
  const axisLabel = (encoding, rotate = 0) => ({ color: "#706A84", fontSize: 11, rotate, hideOverlap: true, formatter: (value) => format(value, encoding, true) });

  function cartesianOption(normalized) {
    const { family, spec, rows, x, y } = normalized;
    if (!x || !y) return null;
    const horizontal = spec.orientation === "HORIZONTAL" && ["BAR", "GROUPED_BAR", "STACKED_BAR"].includes(family);
    const categories = rows.map((row) => valueAt(row, x));
    const many = categories.length > 12;
    const seriesDefs = Array.isArray(spec.series) && spec.series.length > 1 ? spec.series : [{ label: y.semantic_label, measure_field: y.field }];
    const series = seriesDefs.map((definition, index) => ({
      name: definition.label || y.semantic_label || `Series ${index + 1}`,
      type: family === "LINE" || family === "AREA" ? "line" : "bar",
      colorBy: seriesDefs.length === 1 && spec.palette_role === "CATEGORICAL" ? "data" : "series",
      stack: family === "STACKED_BAR" ? (definition.stack_group || "total") : undefined,
      areaStyle: family === "AREA" ? { opacity: 0.16 } : undefined,
      smooth: false,
      symbolSize: 6,
      showSymbol: rows.length <= 36,
      emphasis: { focus: "series" },
      label: { show: spec.labels === "VALUE", position: horizontal ? "right" : "top", color: "#706A84", fontSize: 10, formatter: (item) => format(item.value, y, true) },
      data: rows.map((row) => row[definition.measure_field]),
    }));
    const categoryAxis = { type: "category", data: categories, boundaryGap: !["LINE", "AREA"].includes(family), axisLabel: { ...axisLabel(x, !horizontal && many ? 28 : 0), interval: categories.length <= 12 ? 0 : "auto", width: horizontal ? 120 : 90, overflow: "truncate" }, axisLine: { lineStyle: { color: "#D5D6DF" } }, axisTick: { show: false } };
    const valueAxis = { type: "value", axisLabel: axisLabel(y), splitLine: { lineStyle: { color: "#EFF0F4" } }, axisLine: { show: false }, axisTick: { show: false } };
    return { xAxis: horizontal ? valueAxis : categoryAxis, yAxis: horizontal ? categoryAxis : valueAxis, series };
  }

  function buildFamilyOption(normalized) {
    const { family, spec, rows, x, y } = normalized;
    if (["BAR", "GROUPED_BAR", "STACKED_BAR", "LINE", "AREA"].includes(family)) return cartesianOption(normalized);
    if (family === "SCATTER") {
      if (!x || !y) return null;
      return {
        xAxis: { type: "value", name: x.semantic_label, nameLocation: "middle", nameGap: 30, axisLabel: axisLabel(x), splitLine: { lineStyle: { color: "#EFF0F4" } } },
        yAxis: { type: "value", name: y.semantic_label, nameLocation: "middle", nameGap: 42, axisLabel: axisLabel(y), splitLine: { lineStyle: { color: "#EFF0F4" } } },
        series: [{ name: y.semantic_label, type: "scatter", symbolSize: 8, data: rows.map((row) => [valueAt(row, x), valueAt(row, y)]), emphasis: { focus: "series" } }],
      };
    }
    if (family === "HISTOGRAM") {
      if (!y) return null;
      return {
        xAxis: { type: "category", data: rows.map((row) => `${formatter.formatNumber(row.bin_start)}–${formatter.formatNumber(row.bin_end)}`), axisLabel: { color: "#706A84", fontSize: 10, hideOverlap: true }, axisLine: { lineStyle: { color: "#D5D6DF" } }, axisTick: { show: false } },
        yAxis: { type: "value", axisLabel: axisLabel(y), splitLine: { lineStyle: { color: "#EFF0F4" } } },
        series: [{ name: y.semantic_label || "Count", type: "bar", barCategoryGap: "4%", data: rows.map((row) => row.count) }],
      };
    }
    if (family === "PIE" || family === "DOUGHNUT") {
      if (!x || !y) return null;
      return { series: [{ name: y.semantic_label, type: "pie", radius: family === "DOUGHNUT" ? ["43%", "64%"] : ["0%", "64%"], center: ["50%", "53%"], avoidLabelOverlap: true, minAngle: 2, label: { show: spec.labels !== "NONE", color: "#37324C", formatter: spec.labels === "PERCENT" ? "{b}\n{d}%" : "{b}" }, labelLine: { length: 8, length2: 6 }, data: rows.map((row) => ({ name: valueAt(row, x), value: valueAt(row, y) })) }] };
    }
    if (family === "BOX_PLOT") {
      const required = ["low", "q1", "median", "q3", "high"];
      if (!x || !rows.every((row) => required.every((field) => Number.isFinite(Number(row[field]))))) return null;
      return { xAxis: { type: "category", data: rows.map((row) => valueAt(row, x)), axisLabel: axisLabel(x) }, yAxis: { type: "value", splitLine: { lineStyle: { color: "#EFF0F4" } } }, series: [{ type: "boxplot", data: rows.map((row) => required.map((field) => row[field])) }] };
    }
    if (family === "HEATMAP") {
      const color = normalized.color;
      if (!x || !y || !color) return null;
      const xs = [...new Set(rows.map((row) => valueAt(row, x)))];
      const ys = [...new Set(rows.map((row) => valueAt(row, y)))];
      return { xAxis: { type: "category", data: xs, axisLabel: axisLabel(x) }, yAxis: { type: "category", data: ys, axisLabel: axisLabel(y) }, visualMap: { min: Math.min(...rows.map((row) => Number(valueAt(row, color)))), max: Math.max(...rows.map((row) => Number(valueAt(row, color)))), calculable: false, orient: "horizontal", left: "center", bottom: 0, inRange: { color: theme.palette("SEQUENTIAL") } }, series: [{ type: "heatmap", data: rows.map((row) => [xs.indexOf(valueAt(row, x)), ys.indexOf(valueAt(row, y)), valueAt(row, color)]), label: { show: spec.labels === "VALUE" } }] };
    }
    return null;
  }

  function buildEChartsOption(input) {
    const normalized = normalizeRendererInput(input);
    if (normalized.status !== "READY") return normalized;
    const mapped = buildFamilyOption(normalized);
    if (!mapped) return { ...normalized, status: "UNSUPPORTED_DATA_SHAPE", message: `Verified data does not contain the fields required for ${normalized.family}.` };
    const { spec, rows, x, y } = normalized;
    const zoom = Array.isArray(spec.interaction_hints) && spec.interaction_hints.includes("ZOOM") && rows.length > 12;
    const option = {
      ...theme.BASE,
      color: theme.palette(spec.palette_role),
      aria: { enabled: true, decal: { show: false }, description: `${spec.title}. ${spec.subtitle || ""}`.trim() },
      grid: { top: 20, right: 22, bottom: zoom ? 64 : 44, left: 56, containLabel: true },
      legend: { show: spec.legend === "VISIBLE" || (spec.legend === "AUTO" && ((spec.series || []).length > 1 || ["PIE", "DOUGHNUT"].includes(normalized.family))), type: "scroll", top: 0, right: 0, textStyle: { color: "#706A84", fontSize: 11 } },
      tooltip: {
        show: spec.tooltip !== false, trigger: ["PIE", "DOUGHNUT", "SCATTER"].includes(normalized.family) ? "item" : "axis",
        backgroundColor: "#202238", borderWidth: 0, padding: [9, 11], textStyle: { color: "#FFFFFF", fontSize: 12 },
        formatter(params) {
          const values = Array.isArray(params) ? params : [params];
          return values.map((item) => {
            const raw = Array.isArray(item.value) ? item.value[item.value.length - 1] : (item.value && item.value.value !== undefined ? item.value.value : item.value);
            const name = item.seriesName || (y && y.semantic_label) || "Value";
            const category = item.name != null ? `<div style="opacity:.72;margin-bottom:3px">${escapeHtml(item.name)}</div>` : "";
            return `${category}<strong>${escapeHtml(name)}</strong>: ${escapeHtml(format(raw, y, false))}`;
          }).join("<br>");
        },
      },
      dataZoom: zoom ? [{ type: "inside", filterMode: "none", ...(spec.orientation === "HORIZONTAL" ? { yAxisIndex: 0 } : { xAxisIndex: 0 }) }, { type: "slider", height: 18, bottom: 8, borderColor: "transparent", fillerColor: "rgba(69,111,145,.12)", ...(spec.orientation === "HORIZONTAL" ? { yAxisIndex: 0, orient: "vertical", right: 2, width: 16, height: "70%" } : { xAxisIndex: 0 }) }] : [],
      ...mapped,
    };
    return { status: "READY", family: normalized.family, option };
  }

  function renderState(container, state) {
    container.classList.add("chart-render-state");
    container.textContent = state.message || "This visualization cannot be rendered.";
    container.dataset.rendererStatus = state.status;
  }

  function render(container, input, echartsApi) {
    if (!container) return { status: "INVALID", message: "A chart container is required." };
    const built = buildEChartsOption(input);
    if (built.status !== "READY") { dispose(container); renderState(container, built); return built; }
    const engine = echartsApi || (root && root.echarts);
    if (!engine || typeof engine.init !== "function") return { status: "ECHARTS_UNAVAILABLE", message: "The professional chart renderer could not be loaded." };
    container.classList.remove("chart-render-state");
    container.textContent = "";
    let chart = instances.get(container) || (engine.getInstanceByDom && engine.getInstanceByDom(container));
    if (!chart) chart = engine.init(container, null, { renderer: "canvas", useDirtyRect: true });
    instances.set(container, chart);
    chart.setOption(built.option, { notMerge: true, lazyUpdate: false });
    chart.resize();
    if (typeof ResizeObserver !== "undefined" && !observers.has(container)) {
      const observer = new ResizeObserver(() => { const active = instances.get(container); if (active && !active.isDisposed()) active.resize(); });
      observer.observe(container);
      observers.set(container, observer);
    }
    container.dataset.rendererStatus = "READY";
    return { status: "READY", chart, option: built.option };
  }

  function dispose(container) {
    const observer = observers.get(container);
    if (observer) observer.disconnect();
    observers.delete(container);
    const chart = instances.get(container);
    if (chart && (!chart.isDisposed || !chart.isDisposed())) chart.dispose();
    instances.delete(container);
  }

  function disposeWithin(scope) { if (scope) scope.querySelectorAll("[data-echarts-host]").forEach(dispose); }
  function resizeWithin(scope) { if (scope) scope.querySelectorAll("[data-echarts-host]").forEach((node) => { const chart = instances.get(node); if (chart && (!chart.isDisposed || !chart.isDisposed())) chart.resize(); }); }

  return { SUPPORTED, normalizeRendererInput, buildEChartsOption, render, dispose, disposeWithin, resizeWithin };
});
