const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const main = fs.readFileSync(path.join(root, "main.js"), "utf8");
const css = fs.readFileSync(path.join(root, "style.css"), "utf8");

test("Dataset is a separate primary navigation view", () => {
  assert.match(html, /id="nav-datasets"/);
  assert.match(html, /id="view-dataset"/);
  assert.match(main, /showState\(STATES\.DATASET\)/);
  assert.doesNotMatch(html.match(/id="view-dataset"[\s\S]*?<\/section>\s*<section class="app-view workspace-view"/)[0], /workspace-tabs/);
});

test("Dataset explorer exposes real version, query, grid, and pagination controls", () => {
  for (const id of [
    "dataset-page-name", "dataset-search", "dataset-filter-column", "dataset-filter-value",
    "dataset-sort-column", "dataset-grid-head", "dataset-grid-body", "dataset-row-range",
    "dataset-prev-page", "dataset-next-page", "dataset-page-size",
  ]) assert.match(html, new RegExp(`id="${id}"`));
  assert.match(html, /data-dataset-version="original"/);
  assert.match(html, /data-dataset-version="cleaned"/);
  assert.match(main, /LYNXApiClient\.getDatasetPage\(currentRunId/);
  assert.match(main, /datasetExplorer\.offset/);
});

test("change highlighting remains unavailable without verified cell lineage", () => {
  assert.match(html, /id="dataset-highlight-changes"[^>]*disabled/);
  assert.match(main, /change_highlighting_available/);
  assert.doesNotMatch(css, /dataset-cell-changed/);
});

test("wide tables scroll inside the explorer and retain sticky headers", () => {
  assert.match(css, /\.dataset-grid-scroll \{[^}]*overflow: auto/);
  assert.match(css, /\.dataset-grid thead th \{[^}]*position: sticky/);
  assert.match(css, /\.dataset-grid \{[^}]*width: max-content/);
});
