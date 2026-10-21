# LYNX — UI Redesign Implementation Specification

Status: Approved visual direction; implementation contract for Codex.

## 1. Objective
Replace the current frontend presentation with the approved LYNX designs while preserving working application behavior. This is a frontend/UI migration, **not** a backend rewrite. Implement iteratively, starting with Home, then Overview, Profiling, Prediction, Insights, and Visualization. Do not redesign unapproved Dataset, Export, Setting, or Help pages yet.

## 2. Source of truth and conflict rules
- The approved screenshot supplied with each implementation task is the **visual reference** for that screen: proportions, hierarchy, composition, typography, borders, spacing, and interactions implied by the UI.
- This document governs shared rules, semantic behavior, and scope. Existing live code/APIs govern actual supported functionality and data contracts.
- Screenshot text/numbers/dates/charts are **illustrative**, not real dataset facts. Never hardcode them as live analytics outputs.
- When the screenshot conflicts with an explicit rule here, follow the explicit rule and flag the discrepancy.
- Don't guess unsupported data, endpoints, filenames, component frameworks, routing, or API response shapes; inspect the repository.

## 3. Guardrails
1. Before edits, inspect project structure, package manager, routing, current components, global styles, state management, API usage, and tests; report a succinct inventory and implementation plan.
2. Keep working backend integrations and existing user flows intact. Do not change backend, data pipelines, inference, database schemas, or API contracts without approval.
3. Reuse the frontend framework and dependencies already present. Avoid unnecessary new libraries or duplicated design systems.
4. Preserve route behavior, dataset selection, upload validations, errors, loading states, and frontend keyboard/accessibility behavior wherever supported.
5. For new backend-dependent elements, implement a clearly identified UI-ready state: component, accessible disabled/placeholder behavior, typed integration boundary, and TODO documentation. Never simulate successful predictions, semantically grounded findings, or persisted server actions as if real.
6. Treat blank/loading/error/empty/unavailable states as first-class UI states. Avoid dead buttons: either wire to existing functionality or disable with explanatory text.
7. Implement one screen/task at a time and stop for review. Do not alter unrelated pages or perform speculative cleanup.
8. No glassmorphism, oversized decorative cards, sparkle/diamond/Gemini-style identity marks, or arbitrary animations.

## 4. Navigation and global shell
Sidebar content and order: LYNX; Home, Analysis, Dataset, Export, Setting, Help; Recent Datasets; user profile pinned bottom. Desktop sidebar stays consistent between pages. Sidebar recent list scrolls independently when long, without displacing profile or main navigation.
- Home route: **Home active**; no Analysis tab bar. No recent dataset item highlighted merely because it appears first.
- Analysis routes: **Analysis active**; show exactly **Overview | Profiling | Prediction | Insights | Visualization** in this order. Active tab uses a thin purple underline.
- Recent datasets are dataset navigation shortcuts, not independent analysis tabs. An opened dataset can be indicated within Analysis. Remove from Recents must not imply deleting the underlying dataset.
- Page header under Analysis: dataset display name; short tab-specific subtitle; optional grounded contextual status/timestamp on right.
- On narrow widths, maintain functional navigation; prevent horizontal overflow of page shell. Wide data grids may scroll within themselves.

## 5. Design tokens (starting values; tune to match approved screenshots)
- Font: Inter or current closest equivalent; accessible fallbacks.
- Brand purple: #532BFF; darker interactive text: #3F1FD6; hover: #6B47FF; subtle selection #F1ECFF.
- Page background #F8FAFD; sidebar #FBFCFE; card #FFFFFF; subtle neutral #F6F8FC.
- Text primary #181C3A; secondary #68708C; muted #97A0B8.
- Border #E6EAF2; stronger border #D7DEEA.
- Status: success #1F9D5A / #EAF7F0; warning #E69A1F / #FFF4E1; danger #D85A5A / #FDECEC.
- Spacing units: 4, 8, 12, 16, 20, 24, 32, 40, 48, 64 px.
- Sidebar width target: ~240–250 px. Navigation row height: ~44 px; icons ~18–20 px.
- Content gutters: ~28–32 px desktop (adjust for actual app viewport).
- Radius: inputs/buttons ~12–14 px; cards ~14–18 px; chips pill.
- Borders more important than shadow; normal card shadow none or extremely subtle.
- Approximate type scale: home hero 44–56px (responsive), analysis page title 28–32px, section 18–24px, body 14–16px, caption 12–13px. Use consistent line heights.
- Primary buttons ~44px height, white text on purple, visible hover/focus/disabled states.
- Table headings subtly tinted; numeric values align consistently; dense but readable row heights ~44–48px.
- Charts: restrained purple scale, subdued axes, fine gridlines, real-world units and dates. Do not normalize display values.
- Use clean outline icons. The custom LYNX line-and-node emblem (provided separately) is used for LYNX conversational identity, not sparkles.

## 6. Reusable UI architecture (fit to existing codebase)
AppShell, SidebarNav, RecentDatasetList, UserProfileFooter, AnalysisTabNav, DatasetPageHeader, Card, MetricCard, StatusBadge, Button, Input, DataTable, EmptyState, LoadingState, ErrorState, UploadPanel, RecentDatasetCard, ColumnDetailPanel, PredictionSetup, EvaluationChart, InsightsComposer, VisualizationComposer, ChartExplanationPair.
Build/reuse components only when there is real reuse. Keep business logic outside presentational components.

## 7. Screens

### 7.1 Home — implement first
Purpose: accept a dataset and provide shortcuts to previously opened datasets. No Analysis tab strip.
Approved layout:
- Left shared sidebar with Home active; neutral recent sidebar entries.
- Centered headline **Start with your dataset** and concise subtitle **Upload a CSV or spreadsheet to begin analysis in LYNX.**
- Main focal component is the **illustrated Upload dataset card**, not a dashed generic dropzone: layered purple document/data tiles, heading **Upload dataset**, single compact format/limit line (CSV, XLSX, XLS; up to 50 MB only if confirmed by runtime config), solid **Choose file** button, understated **or** divider, then **You can also drag and drop your file here**.
- Keep an elegant, subtle decorative wave/dot background; decoration must not obscure content or intercept interactions.
- Beneath: understated **Upload → Analyze → Explore** 3-step line.
- Below: compact row of up to four **Recent Datasets** cards, with View all linking to the existing appropriate dataset listing route when available. Each card includes real name and available metadata, no invented values.
- Sidebar recent list and main Home recent cards may both show the same real dataset history; neither is preselected on Home.
- Reuse existing file picker, drop behavior, format/size validation, upload API, progress and failure handling, and success transition. Don't introduce upload endpoints or invent accepted file limits.
- Empty recents should show a tasteful empty state rather than sample datasets.
- Provide visible drag-over, uploading, validation-error, failure, and success states.
- All upload elements must be keyboard accessible, with accessible labels and status announcements.

### 7.2 Analysis / Overview
Use approved Overview illustration: dataset header, four KPI cards **Total Rows, Total Columns, Memory Footprint, Missing Value Percentage**; dataset health badge derived from genuine validation; optionally a data-quality summary only if data exists; raw first rows grid preserving unmodified incoming values. Avoid made-up health scores or detection outcomes.

### 7.3 Analysis / Profiling
Use approved Profiling illustration: summary metrics, dominant searchable/filterable **Semantic Manifest** table with column name, physical data type, semantic role, semantic type, proposed cleaning action, and appropriate status; selected row opens a right-side detail panel containing meaning, counts, observed examples, assigned action, and evidence-grounded decision explanation. Indicate proposed versus applied changes. Unknown roles stay unknown when unavailable. Do not fabricate interpretability text.

### 7.4 Analysis / Prediction
Single combined workspace. User always selects target column; task auto-detected only where existing logic supports it. Top prediction setup stays visible, then run summary, task-dependent evaluation cards, feature importance when supported, classification confusion matrix or regression actual-vs-predicted, and evaluated rows table. Do not treat illustrative 100% metrics as real. Preserve existing prediction API. If feature importance is unavailable, show honest unavailable state; do not manufacture it.

### 7.5 Analysis / Insights
Text-first conversational analyst, **no charts or tables**. Paragraph-form dataset summary (only if grounded), subtle suggested question chips, editorial query/answer flow, LYNX line-node response icon, no chat bubbles/WhatsApp look, fixed bottom question composer. Answer content is grounded in actual supported insights. Preserve backend conversation and clarification interactions. Unsupported summaries should show a useful loading/unavailable state rather than invented prose.

### 7.6 Analysis / Visualization
Automated BI visuals from the cleaned, **unscaled presentation dataset**: a variable number of recommended charts (1–4 if supported), relevant real-data KPI cards, alternating **chart left/explanation right** and **explanation left/chart right** rows, and a custom chart-request conversation/result section. Composer fixed to the bottom of the main content workspace, with safe bottom padding so it never covers chart content. Time series, categorical and distribution charts are conditional on available columns. Explanations must be grounded in underlying results; don't create unsupported model-produced assertions. If auto-generated chart or explanatory backend capability doesn't exist, render honest UI-ready states and TODO boundary, not sample graphs pretending to be live.

## 8. Responsiveness and accessibility
- Target desktop fidelity first against provided reference (roughly 16:9).
- Tablet: reduce columns or stack chart/explanation pairs; side panel may become drawer. Mobile: collapsible sidebar, horizontal/scrollable tabs, single-column content, non-clipped composer, accessible tables.
- Keyboard operability and visible focus rings for upload, dataset cards, tabs, search, selection, and composers.
- Semantically correct buttons/links/labels; statuses not communicated by color alone. Respect reduced-motion preferences. Ensure contrast and readable chart labels.

## 9. Work order and definition of done
Stages: (A) inspect frontend and identify safe integration boundaries; (B) common tokens and shell; (C) Home; (D) Overview; (E) Profiling; (F) Prediction; (G) Insights; (H) Visualization; (I) integrated QA.
For each stage:
- Show changed files and why.
- Show verified functional behaviors; list UI-only placeholders and any necessary TODOs.
- Run existing appropriate formatting, lint, type-check, build, and targeted tests; report actual commands and pass/fail.
- Verify route/sidebar/tab consistency, keyboard navigation, desktop screenshot comparison, smaller viewport behavior, loading/empty/error states.
- Stop and wait for visual review before the next screen.

## 10. First-task constraints
Start by inspecting the repository and implementing **only Home** and minimal necessary shared shell/design tokens. Do not touch the visual content of other Analysis tabs in this first pass. Do not redesign Dataset, Export, Setting or Help. Attach the final approved Home illustration alongside the first Codex task.
