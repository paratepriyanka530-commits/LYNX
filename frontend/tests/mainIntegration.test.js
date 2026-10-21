const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const source = fs.readFileSync(path.resolve(__dirname, "../main.js"), "utf8");
const html = fs.readFileSync(path.resolve(__dirname, "../index.html"), "utf8");

test("real pipeline1_report is passed to the existing report adapter", () => {
  assert.match(source, /adaptCleaningReport\(response\.pipeline1_report/);
});

test("prepared target columns are handed directly to setTargetColumns", () => {
  assert.match(source, /setTargetColumns\(response\.prepared_dataset\.target_columns\)/);
});

test("a successful result transitions to the dataset workspace", () => {
  assert.match(source, /onResult\(response\)\s*\{\s*showCompletedAnalysis\(response\)/);
  assert.match(source, /function showCompletedAnalysis\(response\)[\s\S]*showState\(STATES\.WORKSPACE\)/);
});

test("production target Continue cannot enter fake Pipeline 2", () => {
  assert.match(source, /btn-run-prediction/);
  assert.match(source, /addEventListener\("click", \(event\) => event\.preventDefault\(\)\)/);
  assert.doesNotMatch(source, /P2_PROCESSING|orchestrate_prediction|task_type_resolver/);
});

test("browser loads API/controller modules before main", () => {
  assert.ok(html.indexOf('src="apiClient.js?v=') < html.indexOf('src="main.js?v='));
  assert.ok(html.indexOf('src="pipeline1Controller.js"') < html.indexOf('src="main.js?v='));
});

test("changed frontend assets use explicit cache versions", () => {
  assert.match(html, /src="apiClient\.js\?v=20261018-settings"/);
  assert.match(html, /href="style\.css\?v=20261018-settings"/);
  assert.match(html, /src="visualization\/echartsAdapter\.js\?v=20261017-visualization"/);
  assert.match(html, /src="main\.js\?v=20261018-settings"/);
});

test("Insights landing uses verified summary and suggested questions", () => {
  assert.match(source, /LYNXApiClient\.getInsightsLanding\(requestedDataset\)/);
  assert.match(source, /insightsLanding\.summary_text/);
  assert.match(source, /insightsLanding\.suggested_questions/);
  assert.match(html, /id="insights-summary-copy"/);
  assert.match(html, /id="insights-suggested-questions"/);
  assert.match(source, /renderInsightsLanding\("error"\)/);
});

test("visualization explanations prefer additive XAI without replacing legacy content", () => {
  assert.match(source, /const xai = item\.xai \|\| explanation/);
  assert.match(source, /xai\.what_it_shows \|\| explanation\.what_it_shows/);
  assert.match(source, /xai\.why_this_chart \|\| explanation\.why_this_chart/);
  assert.match(source, /<strong>Limitation:<\/strong>/);
});
