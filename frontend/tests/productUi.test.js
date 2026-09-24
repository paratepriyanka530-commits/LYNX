const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const css = fs.readFileSync(path.join(root, "style.css"), "utf8");
const main = fs.readFileSync(path.join(root, "main.js"), "utf8");

test("approved core palette is exact", () => {
  for (const declaration of [
    "--bg-main: #F8F7F2", "--bg-card: #FFFFFF", "--bg-secondary: #F3F0F8",
    "--text-primary: #160B2E", "--text-secondary: #625D6B", "--text-disabled: #9897A1",
    "--accent-primary: #4E1F6E", "--accent-hover: #3C1755", "--accent-light: #EDE4F2",
  ]) assert.ok(css.includes(declaration), declaration);
});

test("workspace exposes exactly the five approved primary tabs", () => {
  const tabs = [...html.matchAll(/class="workspace-tab(?: active)?"[^>]*data-tab="([^"]+)"/g)].map((match) => match[1]);
  assert.deepEqual(tabs, ["overview", "data-profile", "prediction", "insights", "visualizations"]);
});

test("production markup does not expose internal workflow stages", () => {
  for (const label of ["Dataset Ingestion", "Structural Profiling", "Statistical Analysis", "Semantic Understanding", "Decision Engine", "Pipeline 2", "Model Selector"])
    assert.ok(!html.includes(label), label);
});

test("analysis screen follows the supplied eight-star Lynx reference orientation", () => {
  assert.equal((html.match(/class="star s\d"/g) || []).length, 8);
  assert.equal((html.match(/class="line l\d"/g) || []).length, 7);
  assert.match(html, /Lynx constellation path/);
  assert.match(html, /M10 132 L14 119 L26 113 L33 97 L64 92 L104 59 L113 12 L118 5/);
  assert.doesNotMatch(css, /scaleX\s*\(\s*-1\s*\)/);
  for (let index = 1; index <= 7; index += 1) assert.match(css, new RegExp(`@keyframes connect-${index}`));
  for (let index = 1; index <= 8; index += 1) assert.match(css, new RegExp(`@keyframes activate-${index}`));
});

test("constellation has explicit desktop, tablet, and mobile heights", () => {
  assert.match(css, /\.constellation \{ width: auto; height: 140px;/);
  assert.match(css, /@media \(max-width: 900px\)[\s\S]*\.constellation \{ height: 122px; \}/);
  assert.match(css, /@media \(max-width: 680px\)[\s\S]*\.constellation \{ height: 102px; \}/);
});

test("constellation holds then fades only as one completed figure", () => {
  assert.match(css, /\.constellation-active \{ animation: constellation-whole-fade 5\.6s linear infinite; \}/);
  assert.match(css, /@keyframes constellation-whole-fade \{ 0%, 82% \{ opacity: 1; \} 90%, 100% \{ opacity: 0; \} \}/);
  assert.doesNotMatch(css, /\.constellation-active \.star[^}]*fade/i);
  assert.doesNotMatch(css, /\.constellation-active \.line[^}]*opacity[^}]*animation/i);
});

test("reduced motion renders a static constellation", () => {
  assert.match(css, /@media \(prefers-reduced-motion: reduce\)/);
  assert.match(css, /\.constellation-active \{ display: none; animation: none; \}/);
  assert.match(css, /\.constellation-base path \{ stroke: var\(--accent-primary\); opacity: \.72; \}/);
});

test("real LYNX-head logo replaces the rejected L monogram", () => {
  assert.match(html, /class="brand-logo"[^>]*ChatGPT%20Image%20Sep%207,%202026,%2001_16_53%20PM\.png/);
  assert.doesNotMatch(html, /class="brand-mark"|>L<\/span>/);
  assert.match(css, /\.brand-logo \{/);
});

test("header is opaque and reserves top scroll space", () => {
  assert.match(css, /\.app-header \{[^}]*background: var\(--bg-main\)/);
  assert.doesNotMatch(css, /\.app-header \{[^}]*rgba\(/);
  assert.match(css, /html \{[^}]*scroll-padding-top: 63px/);
  assert.match(css, /\.app-view \{[^}]*scroll-margin-top: 63px/);
});

test("More is mobile-only while secondary actions collapse", () => {
  assert.match(css, /\.more-wrap \{ position: relative; display: none; \}/);
  assert.match(css, /@media \(max-width: 680px\)[\s\S]*\.action-secondary \{ display: none; \}[\s\S]*\.more-wrap \{ display: block; \}/);
  for (const label of ["New Dataset", "Export", "Settings", "Help"]) assert.ok(html.includes(`>${label}<`), label);
});

test("one persistent header exposes state-appropriate actions", () => {
  assert.doesNotMatch(html, /id="global-actions" hidden/);
  assert.match(main, /workspaceOnlyActions = \[byId\("btn-new-dataset"\), byId\("btn-export"\), byId\("more-new"\), byId\("more-export"\)\]/);
  assert.match(main, /action\.hidden = state !== STATES\.WORKSPACE/);
  assert.match(main, /if \(actions\) actions\.hidden = false/);
});

test("scrollbars are visually hidden without disabling tab or list scrolling", () => {
  assert.match(css, /\.workspace-tabs \{[^}]*overflow-x: auto[^}]*scrollbar-width: none/);
  assert.match(css, /\.workspace-tabs::\-webkit-scrollbar \{ display: none; \}/);
  assert.match(css, /\.column-list \{[^}]*overflow-y: auto[^}]*scrollbar-width: none/);
  assert.doesNotMatch(css, /\.workspace-tabs \{[^}]*overflow[^}]*hidden/);
});

test("CSV, XLSX, and XLS are consistently accepted by browser validation", () => {
  assert.match(html, /accept="\.csv,\.xlsx,\.xls"/);
  assert.match(main, /\["\.csv", "\.xlsx", "\.xls"\]/);
});

test("real report populates Overview and exact targets populate Prediction", () => {
  assert.match(main, /renderOverview\(\)/);
  assert.match(main, /currentReport = response\.pipeline1_report/);
  assert.match(main, /setTargetColumns\(response\.prepared_dataset\.target_columns\)/);
});

test("Data Profile supports search and column selection", () => {
  assert.match(html, /id="column-search" type="search"/);
  assert.match(main, /column-search[\s\S]*addEventListener\("input"/);
  assert.match(main, /function selectProfileColumn\(column\)/);
});

test("prediction boundary cannot fabricate execution", () => {
  assert.match(html, /id="btn-run-prediction"[^>]*disabled/);
  assert.doesNotMatch(main, /fetch\([^)]*predict|showState\([^)]*prediction-processing/i);
});

test("no fake progress percentage or progress bar is present", () => {
  assert.doesNotMatch(html, /progress-bar|estimated completion|\d+% complete/i);
  assert.doesNotMatch(main, /progressPercentage|fakeProgress/i);
});

test("long names are constrained without page overflow", () => {
  assert.match(css, /overflow-x: hidden/);
  assert.match(css, /\.dataset-heading h1[\s\S]*text-overflow: ellipsis/);
  assert.match(css, /\.column-option[\s\S]*text-overflow: ellipsis/);
});

test("all requested viewport widths map to intentional responsive layouts", () => {
  assert.match(css, /@media \(max-width: 900px\)/);
  assert.match(css, /@media \(max-width: 680px\)/);
  assert.match(css, /@media \(max-width: 380px\)/);
  for (const width of [320, 375, 430, 768, 1024, 1280, 1440]) {
    const compact = width <= 380;
    const mobile = width <= 680;
    const stackedProfile = width <= 900;
    assert.equal(compact, width === 320 || width === 375);
    assert.equal(mobile, width <= 430);
    assert.equal(stackedProfile, width <= 768);
  }
  assert.match(css, /\.workspace-tabs[\s\S]*overflow-x: auto/);
});

test("empty states and unsupported future features are explicit", () => {
  assert.match(html, /Deeper insights are not available yet/);
  assert.match(html, /Custom chart controls are not connected yet/);
  assert.match(html, /Export is not available yet/);
});

test("icon-only controls and loader have accessible labels", () => {
  for (const label of ["Export report", "Settings", "Help", "More actions", "LYNX constellation analysis indicator"])
    assert.ok(html.includes(`aria-label="${label}"`), label);
  assert.match(html, /class="skip-link"/);
});

test("persistent product shell contains only justified navigation", () => {
  assert.match(html, /class="app-sidebar"/);
  assert.match(html, /id="nav-datasets"/);
  assert.doesNotMatch(html, />Reports<|>Billing<|>Teams<|John Doe|global search/i);
  assert.match(css, /\.app-sidebar \{[\s\S]*position: fixed/);
  assert.match(css, /main \{[^}]*margin-left: var\(--sidebar-width\)/);
});

test("light and dark themes use semantic tokens and persist user choice", () => {
  assert.match(css, /:root\[data-theme="dark"\]/);
  assert.match(css, /--shell-surface: #151419/);
  assert.match(html, /id="btn-theme"/);
  assert.match(html, /localStorage\.getItem\("lynx-theme"\)/);
  assert.match(main, /localStorage\.setItem\("lynx-theme", next\)/);
  assert.match(main, /function toggleTheme\(\)/);
});

test("mobile shell removes the desktop rail rather than squeezing it", () => {
  assert.match(css, /@media \(max-width: 768px\)[\s\S]*\.app-sidebar \{ display: none; \}/);
  assert.match(css, /@media \(max-width: 768px\)[\s\S]*\.mobile-brand \{ display: inline-flex/);
});
