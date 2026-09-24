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
  assert.match(source, /onResult\(response\)[\s\S]*showState\(STATES\.WORKSPACE\)/);
});

test("production target Continue cannot enter fake Pipeline 2", () => {
  assert.match(source, /btn-run-prediction/);
  assert.match(source, /addEventListener\("click", \(event\) => event\.preventDefault\(\)\)/);
  assert.doesNotMatch(source, /P2_PROCESSING|orchestrate_prediction|task_type_resolver/);
});

test("browser loads API/controller modules before main", () => {
  assert.ok(html.indexOf('src="apiClient.js"') < html.indexOf('src="main.js"'));
  assert.ok(html.indexOf('src="pipeline1Controller.js"') < html.indexOf('src="main.js"'));
});
