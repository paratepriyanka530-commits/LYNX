const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const { createVisualizationController } = require("../visualization/visualizationController.js");

test("controller mounts multiple charts through one adapter and disposes on dataset change", () => {
  const hosts = [{ removeAttribute() {}, dataset: {} }, { removeAttribute() {}, dataset: {} }];
  const calls = [];
  const adapter = {
    render(host, visual) { calls.push([host, visual]); return { status: "READY" }; },
    disposeWithin(scope) { calls.push(["dispose", scope]); },
    resizeWithin(scope) { calls.push(["resize", scope]); },
  };
  const scope = { querySelectorAll: () => hosts };
  const controller = createVisualizationController({ adapter });
  assert.deepEqual(controller.renderAll(scope, [{ id: 1 }, { id: 2 }]).map((item) => item.status), ["READY", "READY"]);
  controller.resizeWithin(scope);
  controller.disposeWithin(scope);
  assert.equal(calls.filter((item) => item[0] === "dispose").length, 1);
  assert.equal(calls.filter((item) => item[0] === "resize").length, 1);
});

test("missing ECharts falls back once to deprecated SVG renderer", () => {
  const host = { dataset: {}, removeAttribute(name) { this.removed = name; } };
  const legacy = { render(target, visual) { target.legacy = visual; } };
  const controller = createVisualizationController({
    adapter: { render: () => ({ status: "ECHARTS_UNAVAILABLE" }), disposeWithin() {}, resizeWithin() {} },
    legacyRenderer: legacy,
  });
  const result = controller.renderAll({ querySelectorAll: () => [host] }, [{ chart_spec: {} }]);
  assert.equal(result[0].status, "LEGACY_FALLBACK");
  assert.equal(host.dataset.rendererStatus, "LEGACY_FALLBACK");
});

test("main UI wires dashboard KPIs, chart mounting, clarification surfaces, and cleanup", () => {
  const root = path.resolve(__dirname, "..");
  const main = fs.readFileSync(path.join(root, "main.js"), "utf8");
  const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
  const css = fs.readFileSync(path.join(root, "style.css"), "utf8");
  assert.match(html, /echarts@6\.1\.0/);
  assert.match(html, /id="visualization-kpis"/);
  assert.match(main, /renderVisualizationKpis\(payload\.kpis\)/);
  assert.match(main, /visualizationRenderer\.renderAll\(grid, visualizations\)/);
  assert.match(main, /visualizationRenderer\.disposeWithin\(grid\)/);
  assert.match(main, /visualizationRenderer\.resizeWithin/);
  assert.match(main, /clarification-choices/);
  assert.match(css, /\.echarts-host \{[^}]*height:/);
  assert.match(html, /class="[^"]*visualization-story-list[^"]*"/);
  assert.match(html, /placeholder="Ask LYNX to generate a chart from this dataset\.\.\."/);
  assert.match(main, /explanation-first/);
  assert.match(main, /User-requested visual/);
  assert.match(css, /\.visualization-composer-dock \{[^}]*position: fixed/);
  assert.match(css, /\.visualization-story\.explanation-first/);
});
