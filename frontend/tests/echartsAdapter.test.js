const test = require("node:test");
const assert = require("node:assert/strict");

const adapter = require("../visualization/echartsAdapter.js");
const formatter = require("../visualization/chartFormatter.js");
const theme = require("../visualization/chartTheme.js");

function encoding(channel, field, format_hint = "NUMBER", role = "MEASURE") {
  return { channel, field, semantic_label: field.replaceAll("_", " "), data_type: channel === "X" ? "categorical" : "numeric", role, source_field: null, format_hint };
}

function spec(family, rows, overrides = {}) {
  return {
    chart_id: "chart_test", family, title: `${family} example`, subtitle: null,
    encodings: [encoding("X", "category"), encoding("Y", "value")],
    series: [{ id: "one", label: "Value", measure_field: "value", dimension_filter: null, stack_group: null }],
    orientation: "CARTESIAN", stacking: "NONE", legend: "AUTO", tooltip: true,
    labels: "AUTO", number_format: "NUMBER", palette_role: "SINGLE_SERIES",
    interaction_hints: ["HOVER"],
    data_ref: { columns: Object.keys(rows[0] || {}), rows, coverage: {}, truncated: false, display_limit: null, other_bucket_allowed: false, source_result_id: "result_test" },
    warnings: [], ...overrides,
  };
}

const categoryRows = [{ category: "Alpha", value: 12 }, { category: "Beta", value: 21 }];

test("BAR, GROUPED_BAR, and STACKED_BAR map to bar series", () => {
  for (const family of ["BAR", "GROUPED_BAR", "STACKED_BAR"]) {
    const input = spec(family, categoryRows, family === "STACKED_BAR" ? { stacking: "NORMAL" } : {});
    const result = adapter.buildEChartsOption(input);
    assert.equal(result.status, "READY");
    assert.equal(result.option.series[0].type, "bar");
    assert.equal(Boolean(result.option.series[0].stack), family === "STACKED_BAR");
  }
});

test("LINE and AREA retain ordered values and area semantics", () => {
  const rows = [{ period: "2025-01-01", value: 2 }, { period: "2025-02-01", value: 4 }];
  for (const family of ["LINE", "AREA"]) {
    const result = adapter.buildEChartsOption(spec(family, rows, { encodings: [encoding("X", "period", "DATE", "TEMPORAL"), encoding("Y", "value")] }));
    assert.equal(result.option.series[0].type, "line");
    assert.deepEqual(result.option.xAxis.data, rows.map((row) => row.period));
    assert.equal(Boolean(result.option.series[0].areaStyle), family === "AREA");
  }
});

test("PIE and DOUGHNUT preserve verified name/value pairs", () => {
  for (const family of ["PIE", "DOUGHNUT"]) {
    const result = adapter.buildEChartsOption(spec(family, categoryRows, { labels: "PERCENT", palette_role: "CATEGORICAL" }));
    assert.deepEqual(result.option.series[0].data[0], { name: "Alpha", value: 12 });
    assert.deepEqual(result.option.series[0].radius, family === "DOUGHNUT" ? ["43%", "64%"] : ["0%", "64%"]);
  }
});

test("SCATTER maps two verified numeric fields without aggregation", () => {
  const rows = [{ first: 1, second: 8 }, { first: 2, second: 5 }];
  const result = adapter.buildEChartsOption(spec("SCATTER", rows, { encodings: [encoding("X", "first"), encoding("Y", "second")] }));
  assert.deepEqual(result.option.series[0].data, [[1, 8], [2, 5]]);
});

test("HISTOGRAM uses precomputed verified bins", () => {
  const rows = [{ bin_start: 0, bin_end: 5, count: 3 }, { bin_start: 5, bin_end: 10, count: 7 }];
  const result = adapter.buildEChartsOption(spec("HISTOGRAM", rows, { encodings: [encoding("X", "bin_start"), encoding("Y", "count", "INTEGER", "COUNT")] }));
  assert.deepEqual(result.option.series[0].data, [3, 7]);
  assert.deepEqual(result.option.xAxis.data, ["0–5", "5–10"]);
});

test("BOX_PLOT requires verified five-number summaries", () => {
  const rows = [{ category: "A", low: 1, q1: 2, median: 3, q3: 4, high: 5 }];
  const result = adapter.buildEChartsOption(spec("BOX_PLOT", rows));
  assert.equal(result.status, "READY");
  assert.deepEqual(result.option.series[0].data[0], [1, 2, 3, 4, 5]);
  assert.equal(adapter.buildEChartsOption(spec("BOX_PLOT", categoryRows)).status, "UNSUPPORTED_DATA_SHAPE");
});

test("HEATMAP requires explicit matrix value encoding", () => {
  const rows = [{ col: "A", row: "X", intensity: 3 }, { col: "B", row: "X", intensity: 8 }];
  const result = adapter.buildEChartsOption(spec("HEATMAP", rows, { encodings: [encoding("X", "col"), encoding("Y", "row"), encoding("COLOR", "intensity")] }));
  assert.equal(result.status, "READY");
  assert.deepEqual(result.option.series[0].data[1], [1, 0, 8]);
});

test("long categories use rotation or horizontal orientation and zoom hints", () => {
  const rows = Array.from({ length: 18 }, (_, index) => ({ category: `A very long category ${index}`, value: index }));
  const vertical = adapter.buildEChartsOption(spec("BAR", rows));
  assert.equal(vertical.option.xAxis.axisLabel.rotate, 28);
  const horizontal = adapter.buildEChartsOption(spec("BAR", rows, { orientation: "HORIZONTAL", interaction_hints: ["ZOOM"] }));
  assert.equal(horizontal.option.yAxis.type, "category");
  assert.equal(horizontal.option.dataZoom[0].yAxisIndex, 0);
});

test("number formats are deterministic and currency requires supplied metadata", () => {
  assert.equal(formatter.formatValue(1250, "NUMBER", { compact: true }), "1.25K");
  assert.equal(formatter.formatValue(12.5, "PERCENT"), "12.5%");
  assert.equal(formatter.formatValue(12.5, "CURRENCY"), "12.5");
  assert.match(formatter.formatValue(12.5, "CURRENCY", { currency: "USD" }), /\$12\.50/);
  assert.match(formatter.formatValue("2025-01-01", "DATE"), /2025/);
});

test("multi-series legend and professional palette roles are mapped", () => {
  const rows = [{ category: "A", first: 1, second: 2 }];
  const input = spec("GROUPED_BAR", rows, {
    palette_role: "MULTI_SERIES",
    encodings: [encoding("X", "category"), encoding("Y", "first")],
    series: [{ id: "a", label: "First", measure_field: "first" }, { id: "b", label: "Second", measure_field: "second" }],
  });
  const result = adapter.buildEChartsOption(input);
  assert.equal(result.option.legend.show, true);
  assert.deepEqual(result.option.color, theme.PALETTES.MULTI_SERIES);
  assert.ok(new Set(result.option.color).size > 1);
});

test("single-series categorical bars distinguish categories with the professional palette", () => {
  const result = adapter.buildEChartsOption(spec("BAR", categoryRows, { palette_role: "CATEGORICAL" }));
  assert.equal(result.option.series[0].colorBy, "data");
  assert.deepEqual(result.option.color, theme.PALETTES.CATEGORICAL);
});

test("empty and renderer-unsupported families return controlled states", () => {
  assert.equal(adapter.buildEChartsOption(spec("BAR", [])).status, "EMPTY");
  assert.equal(adapter.buildEChartsOption(spec("SANKEY", categoryRows)).status, "UNSUPPORTED");
});

test("automatic and requested sources produce the same renderer option", () => {
  const chartSpec = spec("BAR", categoryRows);
  const automatic = adapter.buildEChartsOption({ source: "AUTO", chart_spec: chartSpec });
  const requested = adapter.buildEChartsOption({ source: "USER_REQUESTED", chart_spec: chartSpec });
  assert.deepEqual(JSON.parse(JSON.stringify(automatic.option)), JSON.parse(JSON.stringify(requested.option)));
});
