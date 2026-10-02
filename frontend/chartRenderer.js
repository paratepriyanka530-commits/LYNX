/** Accessible dependency-free rendering for Pipeline 3 chart_data. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.LYNXChartRenderer = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const SUPPORTED = new Set(["BAR", "COUNT_BAR", "LINE", "SCATTER", "HISTOGRAM"]);
  const escapeHtml = (value) => String(value == null ? "" : value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  const numeric = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;
  const format = (value) => Number.isFinite(Number(value))
    ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(Number(value)) : String(value ?? "");

  function plotValues(data, type) {
    if (type === "HISTOGRAM") {
      return (Array.isArray(data.bins) ? data.bins : []).map((bin) => ({
        x: `${format(bin.bin_start)}–${format(bin.bin_end)}`, y: numeric(bin.count), rawX: bin.bin_start,
      }));
    }
    return (Array.isArray(data.points) ? data.points : []).map((point) => ({
      x: point.x, y: numeric(point.y), rawX: point.x,
    }));
  }

  function barSvg(values) {
    const width = 720, height = 270, left = 46, right = 14, top = 12, bottom = 58;
    const max = Math.max(1, ...values.map((point) => Math.abs(point.y)));
    const slot = (width - left - right) / Math.max(values.length, 1);
    const barWidth = Math.max(2, Math.min(38, slot * 0.68));
    return `<svg class="data-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Bar chart with ${values.length} values"><line class="chart-axis" x1="${left}" y1="${top}" x2="${left}" y2="${height-bottom}"/><line class="chart-axis" x1="${left}" y1="${height-bottom}" x2="${width-right}" y2="${height-bottom}"/>${values.map((point, index) => {
      const barHeight = Math.max(1, Math.abs(point.y) / max * (height - top - bottom));
      const x = left + slot * index + (slot - barWidth) / 2;
      const y = height - bottom - barHeight;
      const shortLabel = String(point.x).length > 14 ? `${String(point.x).slice(0, 12)}…` : String(point.x);
      return `<g><title>${escapeHtml(point.x)}: ${escapeHtml(format(point.y))}</title><rect class="chart-mark" x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${barWidth.toFixed(2)}" height="${barHeight.toFixed(2)}"/><text class="chart-label" x="${(x + barWidth / 2).toFixed(2)}" y="${height - bottom + 17}" text-anchor="middle">${escapeHtml(shortLabel)}</text></g>`;
    }).join("")}</svg>`;
  }

  function pointSvg(values, type) {
    const width = 720, height = 270, left = 48, right = 18, top = 14, bottom = 42;
    const numericX = values.every((point) => Number.isFinite(Number(point.rawX)));
    const xs = values.map((point, index) => numericX ? numeric(point.rawX) : index);
    const ys = values.map((point) => point.y);
    const minX = Math.min(...xs, 0), maxX = Math.max(...xs, 1), minY = Math.min(...ys, 0), maxY = Math.max(...ys, 1);
    const scaleX = (value) => left + ((value - minX) / (maxX - minX || 1)) * (width - left - right);
    const scaleY = (value) => height - bottom - ((value - minY) / (maxY - minY || 1)) * (height - top - bottom);
    const coordinates = values.map((point, index) => ({ ...point, px: scaleX(xs[index]), py: scaleY(point.y) }));
    const line = type === "LINE" && coordinates.length > 1
      ? `<polyline class="chart-line" points="${coordinates.map((point) => `${point.px.toFixed(2)},${point.py.toFixed(2)}`).join(" ")}"/>` : "";
    return `<svg class="data-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${type === "LINE" ? "Line" : "Scatter"} chart with ${values.length} points"><line class="chart-axis" x1="${left}" y1="${top}" x2="${left}" y2="${height-bottom}"/><line class="chart-axis" x1="${left}" y1="${height-bottom}" x2="${width-right}" y2="${height-bottom}"/>${line}${coordinates.map((point) => `<circle class="chart-point" cx="${point.px.toFixed(2)}" cy="${point.py.toFixed(2)}" r="${type === "SCATTER" ? 3.2 : 3.8}"><title>${escapeHtml(point.x)}: ${escapeHtml(format(point.y))}</title></circle>`).join("")}</svg>`;
  }

  function visualizationMarkup(visualization) {
    const data = visualization && visualization.chart_data ? visualization.chart_data : {};
    const type = String((visualization && visualization.chart_type) || data.chart_type || "").toUpperCase();
    if (!SUPPORTED.has(type)) return `<p class="chart-empty">This chart type is not supported by the current renderer.</p>`;
    const values = plotValues(data, type);
    if (data.has_data === false || !values.length) return `<p class="chart-empty">No usable chart data was returned.</p>`;
    const limited = values.slice(0, type === "SCATTER" || type === "LINE" ? 500 : 50);
    const svg = type === "BAR" || type === "COUNT_BAR" || type === "HISTOGRAM"
      ? barSvg(limited) : pointSvg(limited, type);
    return `<div class="chart-renderer" data-chart-type="${type}">${svg}<div class="chart-axis-caption"><span>${escapeHtml(data.x_column || visualization.x_column || "")}</span>${data.y_column || visualization.y_column ? `<span>${escapeHtml(data.y_column || visualization.y_column)}</span>` : ""}</div></div>`;
  }

  function render(container, visualization) {
    if (!container) return;
    container.innerHTML = visualizationMarkup(visualization);
  }

  return { SUPPORTED, visualizationMarkup, render };
});
