const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const html = fs.readFileSync(path.resolve(__dirname, "../index.html"), "utf8");
const main = fs.readFileSync(path.resolve(__dirname, "../main.js"), "utf8");
const css = fs.readFileSync(path.resolve(__dirname, "../style.css"), "utf8");

test("Settings is a dedicated five-category application view", () => {
  assert.match(html, /id="view-settings"/);
  for (const category of ["appearance", "dataset", "ai", "privacy", "about"]) {
    assert.ok(html.includes(`data-settings-category="${category}"`));
    assert.ok(html.includes(`data-settings-panel="${category}"`));
  }
  assert.doesNotMatch(html, /id="settings-dialog"/);
  assert.match(main, /SETTINGS: "settings"/);
});

test("supported preferences persist without weakening mandatory safeguards", () => {
  for (const key of ["lynx-theme", "lynx-density", "lynx-animations", "lynx-confirm-replace", "lynx-post-upload", "lynx-show-suggestions"]) {
    assert.ok(main.includes(key), key);
  }
  assert.match(html, /privacy protections are always enforced/i);
  assert.match(html, /cannot be disabled/i);
  assert.doesNotMatch(html, /api key|secret|password/i);
  assert.match(css, /:root\[data-density="compact"\]/);
  assert.match(css, /:root\[data-animations="off"\]/);
});

test("clearing the active browser workspace requires explicit confirmation", () => {
  assert.match(main, /window\.confirm\("Clear the active browser workspace\?/);
  assert.match(html, /does not claim immediate deletion of server artifacts/i);
});
