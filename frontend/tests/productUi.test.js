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
    "--bg-main: #F8F9FB", "--bg-card: #FFFFFF", "--bg-secondary: #F7F6FA",
    "--text-primary: #181C3A", "--text-body: #37324C", "--text-secondary: #706A84", "--text-disabled: #9690A5",
    "--accent-primary: #5124B4", "--accent-dark: #3F178C", "--accent-hover: #6333C2",
    "--accent-light: #F0EBF8", "--border: #E7E4ED", "--sidebar-divider: #ECE8F0",
  ]) assert.ok(css.includes(declaration), declaration);
});

test("Home aliases the shared approved color hierarchy", () => {
  for (const declaration of [
    "--home-navy: #070553", "--home-nav-navy: #00005E", "--home-copy: var(--text-secondary)",
    "--home-active: var(--accent-dark)", "--home-purple: var(--accent-primary)", "--home-selection: var(--accent-light)",
    "--home-divider: var(--sidebar-divider)", "--home-card-border: var(--border-soft)", "--home-recent-border: var(--border)",
  ]) assert.ok(css.includes(declaration), declaration);
});

test("Home sidebar uses the approved readable typography hierarchy", () => {
  assert.match(css, /\.sidebar-link \{[^}]*color: var\(--text-primary\);[^}]*font-weight: 500;/);
  assert.match(css, /\.sidebar-link\.active \{[^}]*color: var\(--accent-dark\);/);
  assert.match(css, /\.sidebar-recents h2 \{[^}]*color: var\(--text-primary\);[^}]*font-weight: 600;/);
  assert.match(css, /\.sidebar-recents-empty \{[^}]*color: var\(--text-secondary\);[^}]*font-weight: 500;/);
  assert.match(css, /\.sidebar-profile small \{[^}]*color: var\(--text-secondary\);[^}]*font-weight: 500;/);
});

test("Home follows the approved information architecture", () => {
  const sidebar = html.match(/<aside class="app-sidebar"[\s\S]*?<\/aside>/)[0];
  const labels = [...sidebar.matchAll(/class="sidebar-link[^>]*>[\s\S]*?<span>([^<]+)<\/span><\/button>/g)]
    .map((match) => match[1]);
  assert.deepEqual(labels, ["Home", "Analysis", "Dataset", "Export", "Setting", "Help"]);
  assert.match(html, /id="nav-home"[^>]*aria-current="page"/);
  assert.match(html, /<h1 id="upload-title">Start with your dataset<\/h1>/);
  assert.match(html, /Upload a CSV or spreadsheet to begin analysis in LYNX\./);
  assert.match(html, /class="upload-illustration"/);
  assert.match(html, /id="btn-choose-file"[^>]*>Choose file<\/button>/);
  assert.match(html, /You can also drag and drop your file here/);
  assert.match(html, /No recent datasets yet/);
  assert.doesNotMatch(html, /Customer Transactions|Employee Records|Pizza Sales|Banking Dataset|Healthcare Survey/);
});

test("Home preserves real upload integration and honest unavailable states", () => {
  assert.match(html, /id="dataset-file-input" accept="\.csv,\.xlsx,\.xls"/);
  assert.match(html, /id="upload-dropzone" role="group"/);
  assert.match(html, /id="btn-run-analysis"[^>]*disabled/);
  assert.match(main, /choose\.addEventListener\("click", \(\) => input\.click\(\)\)/);
  assert.match(main, /dropzone\.addEventListener\("drop"/);
  assert.match(html, /id="nav-datasets"[^>]*disabled/);
  assert.match(html, /id="nav-export"[^>]*disabled/);
  assert.match(html, /No account connected/);
});

test("Home SVG artwork cannot fall back to oversized black fills", () => {
  assert.match(html, /id="home-wave-fade"[\s\S]*?<path fill="none" stroke="url\(#home-wave-fade\)"/);
  assert.match(html, /class="document-sheet"[\s\S]*?<path fill="#FAF9FF" stroke="#D8CFFF"/);
  assert.match(html, /class="data-tile data-tile-chart"[\s\S]*?<svg[^>]*fill="none" stroke="#532BFF"/);
  assert.match(css, /\.home-decoration path \{ fill: none; stroke: url\(#home-wave-fade\);/);
  assert.match(css, /\.data-tile svg \{[^}]*fill: none; stroke: var\(--home-purple\)/);
});

test("workspace exposes exactly the five approved primary tabs", () => {
  const tabs = [...html.matchAll(/class="workspace-tab(?: active)?"[^>]*data-tab="([^"]+)"/g)].map((match) => match[1]);
  assert.deepEqual(tabs, ["overview", "data-profile", "prediction", "insights", "visualizations"]);
});

test("processing uses the Analysis shell without selecting a tab", () => {
  assert.match(html, /class="workspace-tabs analysis-tabs processing-tabs"/);
  const processing = html.match(/class="workspace-tabs analysis-tabs processing-tabs"[\s\S]*?<\/nav>/)[0];
  for (const label of ["Overview", "Profiling", "Prediction", "Insights", "Visualization"])
    assert.ok(processing.includes(`>${label}</button>`), label);
  assert.doesNotMatch(processing, /workspace-tab active|aria-selected="true"/);
  assert.equal((processing.match(/disabled/g) || []).length, 5);
});

test("completed Overview exposes evidence-backed dashboard surfaces", () => {
  for (const id of ["overview-metrics", "overview-quality", "overview-health", "overview-raw-preview"])
    assert.ok(html.includes(`id="${id}"`), id);
  assert.match(main, /after\.total_missing_cells/);
  assert.match(main, /dataset_overview/);
  assert.match(main, /memory_footprint_bytes/);
  assert.match(main, /renderRawDataPreview/);
  assert.match(main, /Preview is withheld because privacy clearance did not pass/);
  assert.doesNotMatch(main, /Not reported by analysis API/);
  assert.doesNotMatch(main, /current analysis response does not include raw row values/);
  assert.doesNotMatch(main, /No critical outliers detected|Healthy Dataset/);
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

test("mobile shell retains the existing LYNX logo asset", () => {
  assert.match(html, /class="brand-logo"[^>]*ChatGPT%20Image%20Sep%207,%202026,%2001_16_53%20PM\.png/);
  assert.doesNotMatch(html, /class="brand-mark"/);
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
  assert.match(main, /action\.hidden = !\[STATES\.WORKSPACE, STATES\.DATASET, STATES\.EXPORT\]\.includes\(state\)/);
  assert.match(main, /if \(actions\) actions\.hidden = false/);
});

test("scrollbars are visually hidden without disabling tab or list scrolling", () => {
  assert.match(css, /\.workspace-tabs \{[^}]*overflow-x: auto[^}]*scrollbar-width: none/);
  assert.match(css, /\.workspace-tabs::\-webkit-scrollbar \{ display: none; \}/);
  assert.match(css, /\.manifest-table-scroll \{[^}]*overflow: auto/);
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

test("Profiling exposes the approved semantic manifest and interactions", () => {
  assert.match(html, /id="column-search" type="search"/);
  assert.match(html, /id="semantic-role-filter"/);
  for (const label of ["Column Name", "Physical Type", "Semantic Role", "Semantic Type", "Cleaning Action", "Status"])
    assert.ok(html.includes(`>${label}<`), label);
  assert.match(main, /column-search[\s\S]*addEventListener\("input"/);
  assert.match(main, /semantic-role-filter[\s\S]*addEventListener\("change"/);
  assert.match(main, /function selectProfileColumn\(columnIndex\)/);
  assert.match(main, /PROFILE_PAGE_SIZE = 10/);
  assert.match(main, /data-column-index/);
  assert.match(main, /renderSelectedColumn\(\)/);
});

test("Profiling keeps semantic, decision, and execution states distinct", () => {
  assert.match(main, /requires_review: "Requires Review"/);
  assert.match(main, /proposed: "Proposed", applied: "Applied", skipped: "Skipped"/);
  assert.match(main, /No action required/);
  assert.match(main, /Explanation unavailable/);
  assert.match(main, /Observed examples are withheld by the privacy policy/);
});

test("brand color hierarchy replaces non-semantic blue accents", () => {
  for (const declaration of ["--accent-primary: #5124B4", "--accent-dark: #3F178C", "--accent-light: #F0EBF8", "--accent-muted: #7B58B5"])
    assert.ok(css.includes(declaration), declaration);
  assert.match(css, /\.kpi-icon\.purple, \.kpi-icon\.blue \{ background: var\(--accent-surface\); color: var\(--accent-muted\); \}/);
  assert.doesNotMatch(css, /#2388E8|#E8F3FE/);
});

test("prediction boundary cannot fabricate execution", () => {
  assert.match(html, /id="btn-run-prediction"[^>]*disabled/);
  assert.doesNotMatch(main, /fetch\([^)]*predict|showState\([^)]*prediction-processing/i);
});

test("Prediction uses the approved workspace and keeps target choice explicit", () => {
  for (const id of [
    "prediction-dataset-name", "prediction-target", "prediction-task-type",
    "prediction-status-label", "prediction-feature-importance", "prediction-rows-table",
  ]) assert.ok(html.includes(`id="${id}"`), id);
  assert.match(html, /Prediction workspace for target-based machine learning/);
  assert.match(html, /id="prediction-target"><option value="">Select a target column/);
  assert.match(main, /resolvePredictionTarget\(currentRunId, target\)/);
  assert.match(main, /predictionController\.cancel\(\)/);
  assert.match(main, /renderFeatureImportance\(result\.feature_importance\)/);
  assert.doesNotMatch(main, /prediction-target"\)\.value\s*=/);
});

test("no fake progress percentage or progress bar is present", () => {
  assert.doesNotMatch(html, /progress-bar|estimated completion|\d+% complete/i);
  assert.doesNotMatch(main, /progressPercentage|fakeProgress/i);
});

test("long names are constrained without page overflow", () => {
  assert.match(css, /overflow-x: hidden/);
  assert.match(css, /\.dataset-heading h1[\s\S]*text-overflow: ellipsis/);
  assert.match(css, /\.manifest-column-name \{[^}]*font-weight: 500/);
  assert.match(css, /\.manifest-table th, \.manifest-table td \{[^}]*text-overflow: ellipsis/);
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

test("Insights and Visualizations expose real Pipeline 3 interaction surfaces", () => {
  assert.match(html, /Ask questions and receive dataset-grounded answers in plain language/);
  assert.match(html, /Auto-generated visual analysis powered by the clean prepared dataset/);
  assert.match(html, /class="visualization-composer-dock"/);
  assert.match(html, /id="insights-form"/);
  assert.match(html, /id="visualization-form"/);
  assert.match(html, /id="view-export"/);
});

test("Export workspace exposes authoritative downloads and availability-driven preview", () => {
  for (const id of ["btn-export-dataset", "btn-generate-report", "export-select-all", "export-preview-list"])
    assert.ok(html.includes(`id="${id}"`), id);
  for (const section of ["overview", "profiling", "prediction", "insights", "visualization"])
    assert.ok(html.includes(`data-export-section="${section}"`), section);
  assert.match(main, /getExportAvailability\(currentRunId, exportContext\(\)\)/);
  assert.match(main, /downloadCleanedDataset\(currentRunId, format\)/);
  assert.match(main, /generateAnalysisReport\(currentRunId/);
  assert.match(css, /\.export-preview-status\.unavailable/);
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
  assert.match(main, /saveSetting\("theme", mode\)/);
  assert.match(main, /function toggleTheme\(\)/);
});

test("mobile shell removes the desktop rail rather than squeezing it", () => {
  assert.match(css, /@media \(max-width: 768px\)[\s\S]*\.app-sidebar \{ display: none; \}/);
  assert.match(css, /@media \(max-width: 768px\)[\s\S]*\.mobile-brand \{ display: inline-flex/);
});
