(function () {
  "use strict";

  const STATES = { UPLOAD: "upload", ANALYSIS: "analysis", WORKSPACE: "workspace" };
  const TABS = ["overview", "data-profile", "prediction", "insights", "visualizations"];
  const ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"];

  let currentState = STATES.UPLOAD;
  let selectedFile = null;
  let currentRunId = null;
  let currentRunStatus = null;
  let pipeline1Result = null;
  let currentReport = null;
  let currentDataset = null;
  let controller = null;
  let elapsedStartedAt = null;
  let elapsedTimer = null;
  let profileColumns = [];
  let selectedProfileColumn = null;
  let targetColumns = [];
  let predictionEligible = false;
  let predictionController = null;
  let currentPredictionId = null;
  let currentPredictionResult = null;
  let currentPredictionTarget = null;
  let predictionElapsedStartedAt = null;
  let predictionElapsedTimer = null;
  let predictionRowsOffset = 0;
  const PREDICTION_ROWS_LIMIT = 50;
  let predictionRowsTotal = 0;

  const byId = (id) => document.getElementById(id);
  const number = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;
  const escapeHtml = (value) => String(value == null ? "" : value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  const humanize = (value) => {
    const text = String(value || "").replace(/_/g, " ").trim();
    return text ? text.charAt(0).toUpperCase() + text.slice(1) : "Not available";
  };

  document.addEventListener("DOMContentLoaded", () => {
    setupTheme();
    setupRunController();
    setupUpload();
    setupWorkspaceNavigation();
    setupGlobalActions();
    setupProfileControls();
    setupPredictionBoundary();
    showState(STATES.UPLOAD);
  });

  function showState(state) {
    if (!Object.values(STATES).includes(state)) return;
    currentState = state;
    document.querySelectorAll(".app-view").forEach((view) => {
      const active = view.id === `view-${state}`;
      view.classList.toggle("active", active);
      view.setAttribute("aria-hidden", String(!active));
    });
    const actions = byId("global-actions");
    const workspaceOnlyActions = [byId("btn-new-dataset"), byId("btn-export"), byId("more-new"), byId("more-export")];
    if (actions) actions.hidden = false;
    workspaceOnlyActions.forEach((action) => { if (action) action.hidden = state !== STATES.WORKSPACE; });
    const moreMenu = byId("more-menu");
    const moreButton = byId("btn-more");
    if (moreMenu) moreMenu.hidden = true;
    if (moreButton) moreButton.setAttribute("aria-expanded", "false");
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  function formatBytes(bytes) {
    const value = Number(bytes);
    if (!Number.isFinite(value) || value <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    const power = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
    const amount = value / Math.pow(1024, power);
    return `${amount.toFixed(power ? 1 : 0)} ${units[power]}`;
  }

  function fileValidation(file) {
    if (!file || typeof file !== "object") return { valid: false, message: "Choose a dataset file first." };
    const name = String(file.name || "");
    const extension = name.includes(".") ? name.slice(name.lastIndexOf(".")).toLowerCase() : "";
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      return { valid: false, message: "Supported file types are CSV, XLSX, and XLS." };
    }
    if (file.size === 0) return { valid: false, message: "The selected file is empty." };
    return { valid: true, extension };
  }

  function setUploadBusy(busy) {
    const run = byId("btn-run-analysis");
    const replace = byId("btn-replace-file");
    const input = byId("dataset-file-input");
    if (run) run.disabled = busy || !selectedFile;
    if (replace) replace.disabled = busy;
    if (input) input.disabled = busy;
  }

  function hideUploadError() { if (byId("upload-error-box")) byId("upload-error-box").hidden = true; }
  function showUploadError(message) {
    byId("upload-error-text").textContent = message;
    byId("upload-error-box").hidden = false;
  }

  function resetAnalysisState() {
    if (controller) controller.cancel();
    if (predictionController) predictionController.cancel();
    stopElapsed();
    stopPredictionElapsed();
    currentRunId = null;
    currentRunStatus = null;
    pipeline1Result = null;
    currentReport = null;
    currentDataset = null;
    profileColumns = [];
    selectedProfileColumn = null;
    targetColumns = [];
    predictionEligible = false;
    currentPredictionId = null;
    currentPredictionResult = null;
    currentPredictionTarget = null;
    predictionRowsOffset = 0;
    predictionRowsTotal = 0;
    showPredictionSubView("target");
    setUploadBusy(false);
  }

  function startNewDataset() {
    resetAnalysisState();
    selectedFile = null;
    const input = byId("dataset-file-input");
    if (input) input.value = "";
    byId("selected-file-card").hidden = true;
    hideUploadError();
    showState(STATES.UPLOAD);
  }

  function selectFile(file) {
    resetAnalysisState();
    const validation = fileValidation(file);
    if (!validation.valid) {
      selectedFile = null;
      byId("selected-file-card").hidden = true;
      setUploadBusy(false);
      showUploadError(validation.message);
      return;
    }
    selectedFile = file;
    hideUploadError();
    byId("selected-file-name").textContent = file.name;
    byId("selected-file-name").title = file.name;
    byId("selected-file-format").textContent = validation.extension.slice(1).toUpperCase();
    byId("selected-file-size").textContent = formatBytes(file.size);
    byId("selected-file-card").hidden = false;
    setUploadBusy(false);
  }

  function setupUpload() {
    const dropzone = byId("upload-dropzone");
    const input = byId("dataset-file-input");
    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => event.preventDefault());
    });
    dropzone.addEventListener("dragenter", () => dropzone.classList.add("dragover"));
    dropzone.addEventListener("dragover", () => dropzone.classList.add("dragover"));
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (event) => {
      dropzone.classList.remove("dragover");
      if (event.dataTransfer && event.dataTransfer.files.length) selectFile(event.dataTransfer.files[0]);
    });
    dropzone.addEventListener("click", () => input.click());
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); }
    });
    input.addEventListener("change", () => { if (input.files.length) selectFile(input.files[0]); });
    byId("btn-replace-file").addEventListener("click", () => input.click());
    byId("btn-run-analysis").addEventListener("click", beginAnalysis);
    byId("btn-analysis-retry").addEventListener("click", beginAnalysis);
    byId("btn-analysis-cancel").addEventListener("click", startNewDataset);
  }

  function startElapsed() {
    stopElapsed();
    elapsedStartedAt = Date.now();
    updateElapsed();
    elapsedTimer = window.setInterval(updateElapsed, 1000);
  }
  function stopElapsed() { if (elapsedTimer !== null) window.clearInterval(elapsedTimer); elapsedTimer = null; }
  function updateElapsed() {
    if (elapsedStartedAt === null) return;
    const seconds = Math.max(0, Math.floor((Date.now() - elapsedStartedAt) / 1000));
    const minutes = Math.floor(seconds / 60);
    byId("analysis-elapsed").textContent = `${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")} elapsed`;
  }

  function friendlyStatus(status) {
    if (!status) return "This may take a few minutes.";
    if (status.state === "queued") return "Your dataset is queued and will begin shortly.";
    if (status.state === "running") return "Checking structure, quality, meaning, and preparation needs.";
    if (status.state === "network_retry") return "Connection interrupted. Reconnecting to the analysis…";
    if (status.state === "complete") return "Analysis complete. Preparing your workspace…";
    return "This may take a few minutes.";
  }
  function safeFailureMessage(error) {
    const message = error && error.message ? String(error.message) : "The dataset could not be analyzed.";
    return message.replace(/Pipeline 1/gi, "Analysis");
  }

  function setupRunController() {
    controller = LYNXPipeline1Controller.createPipeline1Controller({
      apiClient: LYNXApiClient,
      pollIntervalMs: 2000,
      onRunAccepted(response) {
        currentRunId = response.run_id;
        currentRunStatus = response;
        byId("analysis-error").hidden = true;
        byId("analysis-filename").textContent = selectedFile.name;
        byId("analysis-filename").title = selectedFile.name;
        byId("analysis-status-message").textContent = friendlyStatus(response);
        startElapsed();
        showState(STATES.ANALYSIS);
      },
      onStatus(status) {
        currentRunStatus = status;
        byId("analysis-status-message").textContent = friendlyStatus(status);
      },
      onResult(response) {
        stopElapsed();
        setUploadBusy(false);
        pipeline1Result = response;
        currentReport = response.pipeline1_report;
        const reportLocation = `/api/runs/${encodeURIComponent(response.run_id)}/pipeline1`;
        currentDataset = LYNXReportAdapter.adaptCleaningReport(response.pipeline1_report, { reportLocation });
        renderWorkspace();
        setTargetColumns(response.prepared_dataset.target_columns);
        showState(STATES.WORKSPACE);
        activateTab("overview");
      },
      onFailure(error, context) {
        stopElapsed();
        setUploadBusy(false);
        const message = safeFailureMessage(error);
        if (context.phase === "create") { showUploadError(message); showState(STATES.UPLOAD); return; }
        byId("analysis-error-text").textContent = message;
        byId("analysis-error").hidden = false;
        byId("analysis-status-message").textContent = "Analysis stopped.";
        showState(STATES.ANALYSIS);
      },
    });
  }

  function beginAnalysis() {
    if (!selectedFile || !controller) return;
    hideUploadError();
    setUploadBusy(true);
    controller.start(selectedFile);
  }

  function setupWorkspaceNavigation() {
    document.querySelectorAll(".workspace-tab").forEach((button) => {
      button.addEventListener("click", () => activateTab(button.dataset.tab));
    });
  }
  function activateTab(tab) {
    if (!TABS.includes(tab)) return;
    document.querySelectorAll(".workspace-tab").forEach((button) => {
      const active = button.dataset.tab === tab;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
      if (active) button.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
    document.querySelectorAll(".tab-panel").forEach((panel) => {
      const active = panel.id === `panel-${tab}`;
      panel.classList.toggle("active", active);
      panel.hidden = !active;
    });
  }

  function setupGlobalActions() {
    byId("btn-new-dataset").addEventListener("click", startNewDataset);
    byId("brand-home").addEventListener("click", () => currentState === STATES.WORKSPACE ? activateTab("overview") : showState(STATES.UPLOAD));
    byId("mobile-brand-home").addEventListener("click", () => currentState === STATES.WORKSPACE ? activateTab("overview") : showState(STATES.UPLOAD));
    byId("nav-datasets").addEventListener("click", () => currentState === STATES.WORKSPACE ? activateTab("overview") : showState(STATES.UPLOAD));
    byId("btn-settings").addEventListener("click", () => byId("settings-dialog").showModal());
    byId("btn-help").addEventListener("click", () => byId("help-dialog").showModal());
    document.querySelectorAll("[data-shell-action]").forEach((button) => {
      button.addEventListener("click", () => byId(`${button.dataset.shellAction}-dialog`).showModal());
    });
    const moreButton = byId("btn-more");
    const menu = byId("more-menu");
    moreButton.addEventListener("click", () => {
      menu.hidden = !menu.hidden;
      moreButton.setAttribute("aria-expanded", String(!menu.hidden));
    });
    menu.addEventListener("click", (event) => {
      const action = event.target.dataset.moreAction;
      if (action === "settings") byId("settings-dialog").showModal();
      if (action === "help") byId("help-dialog").showModal();
      if (action === "new") startNewDataset();
      if (action === "theme") toggleTheme();
      menu.hidden = true;
      moreButton.setAttribute("aria-expanded", "false");
    });
    document.addEventListener("click", (event) => {
      if (!event.target.closest(".more-wrap")) { menu.hidden = true; moreButton.setAttribute("aria-expanded", "false"); }
    });
  }

  function currentTheme() {
    return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  }

  function applyTheme(theme, persist = true) {
    const next = theme === "dark" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    if (persist) {
      try { window.localStorage.setItem("lynx-theme", next); } catch (_error) { /* storage may be unavailable */ }
    }
    const toggle = byId("btn-theme");
    const label = byId("theme-label");
    if (toggle) {
      toggle.setAttribute("aria-pressed", String(next === "dark"));
      toggle.setAttribute("aria-label", `Switch to ${next === "dark" ? "light" : "dark"} theme`);
    }
    if (label) label.textContent = next === "dark" ? "Light" : "Dark";
    document.querySelectorAll("[data-theme-choice]").forEach((choice) => {
      const selected = choice.dataset.themeChoice === next;
      choice.classList.toggle("active", selected);
      choice.setAttribute("aria-pressed", String(selected));
    });
  }

  function toggleTheme() { applyTheme(currentTheme() === "dark" ? "light" : "dark"); }

  function setupTheme() {
    applyTheme(currentTheme(), false);
    byId("btn-theme").addEventListener("click", toggleTheme);
    document.querySelectorAll("[data-theme-choice]").forEach((choice) => {
      choice.addEventListener("click", () => applyTheme(choice.dataset.themeChoice));
    });
  }

  function renderWorkspace() {
    const rows = number(pipeline1Result.prepared_dataset.rows);
    const columns = number(pipeline1Result.prepared_dataset.columns);
    byId("workspace-dataset-name").textContent = currentDataset.name;
    byId("workspace-dataset-name").title = currentDataset.name;
    const extension = String(currentDataset.name || "").split(".").pop();
    byId("workspace-file-type").textContent = extension && extension !== currentDataset.name ? extension.toUpperCase() : "DATA";
    byId("workspace-dataset-meta").textContent = `${currentDataset.statusText} · ${rows.toLocaleString()} rows · ${columns.toLocaleString()} columns`;
    renderOverview();
    initializeDataProfile();
    renderInsights();
    renderVisualizations();
  }

  function renderOverview() {
    const report = currentReport;
    const metrics = currentDataset.metrics;
    const validation = report.validation || {};
    const context = (report.semantic_profile && report.semantic_profile.dataset_context) || {};
    byId("overview-summary").textContent = currentDataset.summary || "Analysis completed using the evidence available in the dataset.";
    const status = validation.passed ? "Prepared and validated" : humanize(report.status);
    byId("overview-metrics").innerHTML = [
      ["Rows", metrics.rows.toLocaleString(), ""], ["Columns", metrics.columns.toLocaleString(), ""],
      ["Detected issues", metrics.issuesCount.toLocaleString(), ""], ["Status", status, "status-value"],
    ].map(([label, value, cls]) => `<div class="metric-item"><span>${escapeHtml(label)}</span><strong class="${cls}">${escapeHtml(value)}</strong></div>`).join("");

    const issues = currentDataset.allIssues.slice(0, 6);
    byId("overview-quality").innerHTML = issues.length ? `<ul class="issue-list">${issues.map((issue) => `<li class="issue-row"><div><strong>${escapeHtml(issue.name)}</strong><small>${issue.affectedCount.toLocaleString()} affected · ${issue.percentage.toFixed(2)}%</small></div><span class="severity">${escapeHtml(issue.riskHint)}</span></li>`).join("")}</ul>` : `<p class="empty-inline">No quality issues were detected.</p>`;

    const findings = [];
    if (context.probable_domain) findings.push(`Probable domain: ${context.probable_domain}`);
    if (context.row_entity && context.row_entity !== "unknown") findings.push(`Each row represents: ${context.row_entity}`);
    if (context.dataset_description) findings.push(context.dataset_description);
    (Array.isArray(report.warnings) ? report.warnings : []).slice(0, 3).forEach((warning) => findings.push(typeof warning === "string" ? warning : warning.message));
    byId("overview-findings").innerHTML = findings.filter(Boolean).length ? `<ul class="plain-list">${findings.filter(Boolean).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : `<p class="empty-inline">No additional dataset-level findings were available.</p>`;

    const changed = (Array.isArray(report.executions) ? report.executions : []).filter((execution) => execution.status === "success" && execution.changed);
    const changedCells = number(validation.changed_cell_count);
    const preparation = changed.slice(0, 6).map((execution) => `${humanize(execution.action_id)}${execution.target_columns && execution.target_columns.length ? ` in ${execution.target_columns.join(", ")}` : ""}`);
    if (changedCells) preparation.unshift(`${changedCells.toLocaleString()} values changed by approved actions`);
    byId("overview-preparation").innerHTML = preparation.length ? `<ul class="plain-list">${preparation.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : `<p class="empty-inline">No automatic value changes were required.</p>`;

    const privacyPassed = pipeline1Result.privacy_clearance_passed === true;
    byId("overview-privacy").innerHTML = `<div class="status-line"><span class="status-dot ${privacyPassed ? "" : "warning"}"></span><span>${privacyPassed ? "Privacy clearance passed for the prepared dataset." : "Privacy clearance did not pass. Prediction remains unavailable."}</span></div>`;
  }

  function reportColumnNames() {
    const before = currentReport.before_profile || {};
    if (Array.isArray(before.column_names) && before.column_names.length) return before.column_names.map(String);
    return (before.column_profiles || []).map((column) => String(column.column_name));
  }

  function initializeDataProfile() {
    profileColumns = reportColumnNames();
    selectedProfileColumn = profileColumns[0] || null;
    const select = byId("profile-column-select");
    select.innerHTML = profileColumns.map((column) => `<option value="${escapeHtml(column)}">${escapeHtml(column)}</option>`).join("");
    renderColumnList(profileColumns);
    renderSelectedColumn();
  }

  function setupProfileControls() {
    byId("column-search").addEventListener("input", (event) => {
      const query = event.target.value.trim().toLocaleLowerCase();
      renderColumnList(profileColumns.filter((column) => column.toLocaleLowerCase().includes(query)));
    });
    byId("profile-column-select").addEventListener("change", (event) => selectProfileColumn(event.target.value));
  }

  function renderColumnList(columns) {
    byId("column-count").textContent = `${columns.length.toLocaleString()} of ${profileColumns.length.toLocaleString()} columns`;
    byId("column-list").innerHTML = columns.length ? columns.map((column) => `<button class="column-option ${column === selectedProfileColumn ? "active" : ""}" type="button" role="option" aria-selected="${column === selectedProfileColumn}" data-column="${escapeHtml(column)}" title="${escapeHtml(column)}">${escapeHtml(column)}</button>`).join("") : `<p class="empty-inline">No matching columns.</p>`;
    byId("column-list").querySelectorAll(".column-option").forEach((button) => button.addEventListener("click", () => selectProfileColumn(button.dataset.column)));
  }

  function selectProfileColumn(column) {
    if (!profileColumns.includes(column)) return;
    selectedProfileColumn = column;
    byId("profile-column-select").value = column;
    renderColumnList(profileColumns.filter((name) => name.toLocaleLowerCase().includes(byId("column-search").value.trim().toLocaleLowerCase())));
    renderSelectedColumn();
  }

  function renderSelectedColumn() {
    if (!selectedProfileColumn) {
      byId("column-profile").innerHTML = `<p class="empty-inline">No columns are available.</p>`;
      byId("column-findings").innerHTML = `<p class="empty-inline">No findings are available.</p>`;
      return;
    }
    const profiles = ((currentReport.before_profile || {}).column_profiles) || [];
    const semantics = ((currentReport.semantic_profile || {}).columns) || [];
    const profile = profiles.find((item) => item.column_name === selectedProfileColumn) || {};
    const semantic = semantics.find((item) => item.column_name === selectedProfileColumn) || {};
    const missing = profile.missing || {};
    const unique = profile.uniqueness || {};
    const stats = profile.numeric_statistics || profile.statistics || {};
    const scalarStats = Object.entries(stats).filter(([, value]) => ["string", "number", "boolean"].includes(typeof value)).slice(0, 6);
    const statItems = [
      ["Physical type", profile.physical_dtype || "Not available"], ["Semantic type", semantic.semantic_type || "Not available"],
      ["Semantic role", semantic.semantic_role || "Not available"], ["Non-null values", number(profile.non_null_count).toLocaleString()],
      ["Missing", `${number(missing.count).toLocaleString()} (${number(missing.percentage).toFixed(2)}%)`], ["Unique values", number(unique.unique_count).toLocaleString()],
      ...scalarStats.map(([key, value]) => [humanize(key), String(value)]),
    ];
    const samples = Array.isArray(profile.sample_values) ? profile.sample_values.slice(0, 8) : [];
    byId("column-profile").innerHTML = `<div class="column-title-row"><div><p class="eyebrow">Selected column</p><h2 title="${escapeHtml(selectedProfileColumn)}">${escapeHtml(selectedProfileColumn)}</h2></div><span class="type-tag">${escapeHtml(profile.physical_dtype || "Unknown type")}</span></div><div class="profile-definition"><h3>Meaning</h3><p>${escapeHtml(semantic.concept_description || "No semantic description was available for this column.")}</p></div><div class="profile-stats">${statItems.map(([label, value]) => `<div class="profile-stat"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}</div>${samples.length ? `<div class="sample-block"><h3>Observed examples</h3><div class="sample-values">${samples.map((value) => `<code title="${escapeHtml(value)}">${escapeHtml(value)}</code>`).join("")}</div></div>` : ""}`;

    const issues = (currentReport.issues || []).filter((item) => Array.isArray(item.target_columns) && item.target_columns.includes(selectedProfileColumn));
    const decisions = (currentReport.decisions || []).filter((item) => Array.isArray(item.target_columns) && item.target_columns.includes(selectedProfileColumn));
    const findings = [
      ...issues.map((issue) => ({ title: humanize(issue.issue_type), detail: `${number(issue.affected_count).toLocaleString()} affected (${number(issue.affected_percentage).toFixed(2)}%)` })),
      ...decisions.map((decision) => ({ title: humanize(decision.proposed_action || decision.decision_type), detail: decision.rationale || "Preparation decision recorded." })),
    ];
    byId("column-findings").innerHTML = findings.length ? findings.slice(0, 12).map((finding) => `<div class="finding"><strong>${escapeHtml(finding.title)}</strong><span>${escapeHtml(finding.detail)}</span></div>`).join("") : `<p class="empty-inline">No column-specific issues or changes were recorded.</p>`;
  }

  function setTargetColumns(columns) {
    targetColumns = Array.isArray(columns) ? columns.map(String) : [];
    const select = byId("prediction-target");
    select.innerHTML = `<option value="">Select a target column</option>${targetColumns.map((column) => `<option value="${escapeHtml(column)}">${escapeHtml(column)}</option>`).join("")}`;
    const availability = LYNXPipeline1Controller.eligibilityMessage(pipeline1Result);
    predictionEligible = availability.enabled;
    byId("prediction-eligibility").textContent = availability.enabled
      ? "Select a target column to predict."
      : availability.message;
    byId("prediction-target").disabled = !targetColumns.length || !availability.enabled;
    byId("btn-run-prediction").disabled = true;
    byId("btn-run-prediction").title = availability.enabled
      ? "Select a target column to run prediction."
      : availability.message;
    showPredictionSubView("target");
  }

  function setupPredictionBoundary() {
    setupPredictionController();

    const select = byId("prediction-target");
    const runBtn = byId("btn-run-prediction");

    select.addEventListener("change", () => {
      const selected = select.value;
      currentPredictionTarget = selected;
      if (!selected) {
        runBtn.disabled = true;
        byId("prediction-eligibility").textContent = predictionEligible
          ? "Select a target column to predict."
          : LYNXPipeline1Controller.eligibilityMessage(pipeline1Result).message;
      } else if (!predictionEligible) {
        runBtn.disabled = true;
        byId("prediction-eligibility").textContent = LYNXPipeline1Controller.eligibilityMessage(pipeline1Result).message;
      } else {
        runBtn.disabled = false;
        byId("prediction-eligibility").textContent = "Target selected. Ready to run prediction.";
      }
    });

    runBtn.addEventListener("click", (event) => event.preventDefault());
    runBtn.addEventListener("click", beginPrediction);

    const runAnotherBtn = byId("btn-prediction-run-another");
    if (runAnotherBtn) {
      runAnotherBtn.addEventListener("click", () => {
        showPredictionSubView("target");
      });
    }

    const retryBtn = byId("btn-prediction-retry");
    if (retryBtn) {
      retryBtn.addEventListener("click", () => {
        showPredictionSubView("target");
      });
    }

    const prevPageBtn = byId("btn-pred-prev-page");
    if (prevPageBtn) {
      prevPageBtn.addEventListener("click", () => {
        const targetOffset = Math.max(0, predictionRowsOffset - PREDICTION_ROWS_LIMIT);
        loadPredictionRowsPage(targetOffset);
      });
    }

    const nextPageBtn = byId("btn-pred-next-page");
    if (nextPageBtn) {
      nextPageBtn.addEventListener("click", () => {
        const targetOffset = predictionRowsOffset + PREDICTION_ROWS_LIMIT;
        loadPredictionRowsPage(targetOffset);
      });
    }
  }

  function beginPrediction() {
    const target = byId("prediction-target").value;
    if (!target || !predictionEligible || !currentRunId || !predictionController) return;
    currentPredictionTarget = target;
    byId("prediction-waiting-target-name").textContent = target;
    byId("prediction-target").disabled = true;
    byId("btn-run-prediction").disabled = true;
    showPredictionSubView("waiting");
    predictionController.start(currentRunId, target);
  }

  function setupPredictionController() {
    if (typeof LYNXPredictionController === "undefined") return;
    predictionController = LYNXPredictionController.createPredictionController({
      apiClient: LYNXApiClient,
      pollIntervalMs: 2000,
      onJobCreated(response) {
        currentPredictionId = response.prediction_id;
        showPredictionSubView("waiting");
      },
      onStatus() {
        // Minimal neutral presentation maintained
      },
      onResult(result, initialRows) {
        stopPredictionElapsed();
        currentPredictionResult = result;
        const targetSelect = byId("prediction-target");
        if (targetSelect) targetSelect.disabled = false;
        renderPredictionResults(result, initialRows);
        showPredictionSubView("results");
      },
      onFailure(error) {
        stopPredictionElapsed();
        const targetSelect = byId("prediction-target");
        if (targetSelect) targetSelect.disabled = false;
        const message = error && error.message ? error.message : "Prediction could not be completed.";
        byId("prediction-error-text").textContent = message;
        showPredictionSubView("error");
      },
    });
  }

  function showPredictionSubView(view) {
    const targetSection = byId("prediction-target-section");
    const waitingSection = byId("prediction-waiting-section");
    const resultsSection = byId("prediction-results-section");
    const errorSection = byId("prediction-error-section");

    if (targetSection) targetSection.hidden = view !== "target";
    if (waitingSection) waitingSection.hidden = view !== "waiting";
    if (resultsSection) resultsSection.hidden = view !== "results";
    if (errorSection) errorSection.hidden = view !== "error";

    if (view === "waiting") {
      const holder = byId("prediction-constellation-holder");
      if (holder && !holder.querySelector(".constellation")) {
        const existing = document.querySelector("#view-analysis .constellation");
        if (existing) {
          const clone = existing.cloneNode(true);
          clone.setAttribute("aria-label", "LYNX constellation prediction indicator");
          holder.innerHTML = "";
          holder.appendChild(clone);
        }
      }
      startPredictionElapsed();
    } else {
      stopPredictionElapsed();
    }
  }

  function startPredictionElapsed() {
    stopPredictionElapsed();
    predictionElapsedStartedAt = Date.now();
    updatePredictionElapsed();
    predictionElapsedTimer = window.setInterval(updatePredictionElapsed, 1000);
  }
  function stopPredictionElapsed() {
    if (predictionElapsedTimer !== null) window.clearInterval(predictionElapsedTimer);
    predictionElapsedTimer = null;
  }
  function updatePredictionElapsed() {
    if (predictionElapsedStartedAt === null) return;
    const seconds = Math.max(0, Math.floor((Date.now() - predictionElapsedStartedAt) / 1000));
    const minutes = Math.floor(seconds / 60);
    const text = `${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")} elapsed`;
    if (byId("prediction-elapsed")) byId("prediction-elapsed").textContent = text;
  }

  function formatPercent(value) {
    if (value == null || value === "") return "—";
    const num = Number(value);
    if (!Number.isFinite(num)) return "—";
    return `${(num * 100).toFixed(2)}%`;
  }

  function formatDecimal(value, digits = 4) {
    if (value == null || value === "") return "—";
    const num = Number(value);
    if (!Number.isFinite(num)) return "—";
    return num.toFixed(digits);
  }

  function formatAdaptiveNumber(value) {
    if (value == null) return "—";
    const num = Number(value);
    if (!Number.isFinite(num)) return "—";
    if (Number.isInteger(num)) return num.toLocaleString();
    const abs = Math.abs(num);
    if (abs >= 1000) return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
    if (abs >= 1) return num.toFixed(2);
    if (abs >= 0.001) return num.toFixed(4);
    return num.toPrecision(3);
  }

  function formatConfidence(value) {
    if (value == null) return "—";
    const num = Number(value);
    if (!Number.isFinite(num)) return "—";
    return `${(num * 100).toFixed(2)}%`;
  }

  function renderPredictionResults(result, initialRows) {
    byId("pred-summary-target").textContent = result.target || currentPredictionTarget || "—";
    byId("pred-summary-task").textContent = humanize(result.task_type);

    const testRowsCount = result.predictions && Number.isFinite(result.predictions.row_count)
      ? result.predictions.row_count
      : (result.dataset && Number.isFinite(result.dataset.test_rows)
          ? result.dataset.test_rows
          : (initialRows && Number.isFinite(initialRows.total) ? initialRows.total : "Not available"));
    byId("pred-summary-rows").textContent = Number.isFinite(testRowsCount) ? testRowsCount.toLocaleString() : testRowsCount;

    const isClassification = result.task_type === "classification";
    const metricsContainer = byId("prediction-primary-metrics");
    const secondaryWrapper = byId("prediction-secondary-metrics-wrapper");

    if (isClassification) {
      const primary = (result.metrics && result.metrics.primary) || {};
      const secondary = (result.metrics && result.metrics.secondary) || {};
      const calibration = (result.metrics && result.metrics.calibration) || {};

      metricsContainer.innerHTML = [
        ["Balanced Accuracy", formatPercent(primary.balanced_accuracy)],
        ["F1 Score", formatPercent(primary.f1_macro)],
        ["ROC AUC", formatPercent(primary.roc_auc)],
        ["Accuracy", formatPercent(primary.accuracy)],
      ].map(([label, val]) => `
        <div class="metric-card">
          <span class="metric-card-label">${escapeHtml(label)}</span>
          <strong class="metric-card-value">${escapeHtml(val)}</strong>
        </div>
      `).join("");

      const secondaryItems = [];
      if (secondary.precision_macro != null) secondaryItems.push(["Precision", formatPercent(secondary.precision_macro)]);
      if (secondary.recall_macro != null) secondaryItems.push(["Recall", formatPercent(secondary.recall_macro)]);
      if (secondary.mcc != null) secondaryItems.push(["MCC", formatDecimal(secondary.mcc, 4)]);
      if (secondary.log_loss != null) secondaryItems.push(["Log Loss", formatDecimal(secondary.log_loss, 4)]);
      if (calibration.brier_score != null) secondaryItems.push(["Brier Score", formatDecimal(calibration.brier_score, 4)]);
      if (calibration.ece != null) secondaryItems.push(["ECE", formatPercent(calibration.ece)]);

      if (secondaryItems.length) {
        secondaryWrapper.innerHTML = `
          <details class="secondary-metrics-details">
            <summary>Secondary metrics</summary>
            <div class="secondary-metrics-grid">
              ${secondaryItems.map(([label, val]) => `
                <div class="secondary-metric-item">
                  <span>${escapeHtml(label)}</span>
                  <strong>${escapeHtml(val)}</strong>
                </div>
              `).join("")}
            </div>
          </details>
        `;
      } else {
        secondaryWrapper.innerHTML = "";
      }

      const confusion = result.metrics && result.metrics.confusion_matrix;
      const vizSection = byId("prediction-viz-section");
      const vizContainer = byId("prediction-viz-container");
      if (Array.isArray(confusion) && confusion.length && Array.isArray(confusion[0])) {
        byId("prediction-viz-title").textContent = "Confusion matrix";
        vizSection.hidden = false;
        vizContainer.innerHTML = renderConfusionMatrix(confusion);
      } else {
        vizSection.hidden = true;
      }
    } else {
      const metrics = result.metrics || {};
      metricsContainer.innerHTML = [
        ["R²", formatDecimal(metrics.r2, 4)],
        ["RMSE", formatAdaptiveNumber(metrics.rmse)],
        ["MAE", formatAdaptiveNumber(metrics.mae)],
        ["NRMSE", formatAdaptiveNumber(metrics.nrmse)],
      ].map(([label, val]) => `
        <div class="metric-card">
          <span class="metric-card-label">${escapeHtml(label)}</span>
          <strong class="metric-card-value">${escapeHtml(val)}</strong>
        </div>
      `).join("");
      secondaryWrapper.innerHTML = "";

      const vizSection = byId("prediction-viz-section");
      const vizContainer = byId("prediction-viz-container");
      const rows = initialRows && Array.isArray(initialRows.rows) ? initialRows.rows : [];
      if (rows.length) {
        byId("prediction-viz-title").textContent = "Actual vs. predicted";
        vizSection.hidden = false;
        vizContainer.innerHTML = renderRegressionScatterPlot(rows);
      } else {
        vizSection.hidden = true;
      }
    }

    renderPredictionTable(isClassification, initialRows);
  }

  function renderConfusionMatrix(matrix) {
    const numRows = matrix.length;
    const numCols = matrix[0].length;
    let html = `<div class="confusion-matrix-wrapper"><table class="confusion-matrix-table">`;
    html += `<caption>Confusion Matrix (Rows: Actual, Columns: Predicted)</caption>`;
    html += `<thead><tr><th scope="col">Actual \\ Predicted</th>`;
    for (let c = 0; c < numCols; c++) {
      html += `<th scope="col">Predicted ${c}</th>`;
    }
    html += `</tr></thead><tbody>`;

    for (let r = 0; r < numRows; r++) {
      html += `<tr><th scope="row">Actual ${r}</th>`;
      for (let c = 0; c < numCols; c++) {
        const cell = matrix[r][c];
        const isDiag = r === c;
        html += `<td class="cm-cell ${isDiag ? "cm-highlight" : ""}">${escapeHtml(cell)}</td>`;
      }
      html += `</tr>`;
    }

    html += `</tbody></table></div>`;
    return html;
  }

  function renderRegressionScatterPlot(rows) {
    const validPoints = rows
      .filter((r) => r.actual != null && r.prediction != null && Number.isFinite(Number(r.actual)) && Number.isFinite(Number(r.prediction)))
      .map((r) => ({ actual: Number(r.actual), prediction: Number(r.prediction) }));

    if (!validPoints.length) {
      return `<p class="empty-inline">Insufficient valid row data for scatter plot.</p>`;
    }

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    validPoints.forEach((pt) => {
      if (pt.actual < minX) minX = pt.actual;
      if (pt.actual > maxX) maxX = pt.actual;
      if (pt.prediction < minY) minY = pt.prediction;
      if (pt.prediction > maxY) maxY = pt.prediction;
    });

    const globalMin = Math.min(minX, minY);
    const globalMax = Math.max(maxX, maxY);
    const range = globalMax - globalMin || 1;
    const padMin = globalMin - range * 0.05;
    const padMax = globalMax + range * 0.05;
    const span = padMax - padMin || 1;

    const width = 420;
    const height = 260;
    const ml = 50, mr = 20, mt = 20, mb = 40;
    const plotW = width - ml - mr;
    const plotH = height - mt - mb;

    const scaleX = (val) => ml + ((val - padMin) / span) * plotW;
    const scaleY = (val) => mt + plotH - ((val - padMin) / span) * plotH;

    const lineX1 = scaleX(padMin);
    const lineY1 = scaleY(padMin);
    const lineX2 = scaleX(padMax);
    const lineY2 = scaleY(padMax);

    const dots = validPoints.map((pt) => {
      const cx = scaleX(pt.actual).toFixed(1);
      const cy = scaleY(pt.prediction).toFixed(1);
      return `<circle cx="${cx}" cy="${cy}" r="3.5" fill="var(--accent-primary)" opacity="0.75"><title>Actual: ${pt.actual}, Predicted: ${pt.prediction}</title></circle>`;
    }).join("");

    return `
      <div class="scatter-plot-wrapper">
        <svg class="scatter-plot-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Actual versus predicted scatter plot">
          <title>Actual versus predicted scatter plot</title>
          <line x1="${ml}" y1="${mt}" x2="${ml}" y2="${mt + plotH}" stroke="var(--border)" stroke-width="1"/>
          <line x1="${ml}" y1="${mt + plotH}" x2="${ml + plotW}" y2="${mt + plotH}" stroke="var(--border)" stroke-width="1"/>
          <line x1="${lineX1}" y1="${lineY1}" x2="${lineX2}" y2="${lineY2}" stroke="var(--text-disabled)" stroke-width="1.5" stroke-dasharray="4 4"/>
          ${dots}
          <text x="${ml + plotW / 2}" y="${height - 8}" text-anchor="middle" font-size="11" fill="var(--text-secondary)" font-family="var(--font)">Actual value</text>
          <text x="14" y="${mt + plotH / 2}" text-anchor="middle" font-size="11" fill="var(--text-secondary)" font-family="var(--font)" transform="rotate(-90 14 ${mt + plotH / 2})">Predicted value</text>
        </svg>
      </div>
    `;
  }

  function renderPredictionTable(isClassification, rowsResponse) {
    const head = byId("prediction-table-head");
    const body = byId("prediction-table-body");
    const prevBtn = byId("btn-pred-prev-page");
    const nextBtn = byId("btn-pred-next-page");
    const info = byId("pred-page-info");

    const offset = (rowsResponse && Number(rowsResponse.offset)) || 0;
    const limit = (rowsResponse && Number(rowsResponse.limit)) || PREDICTION_ROWS_LIMIT;
    const total = (rowsResponse && Number(rowsResponse.total)) || 0;
    const rows = (rowsResponse && Array.isArray(rowsResponse.rows)) ? rowsResponse.rows : [];

    predictionRowsOffset = offset;
    predictionRowsTotal = total;

    if (isClassification) {
      head.innerHTML = `<tr><th scope="col">#</th><th scope="col">Actual</th><th scope="col">Prediction</th><th scope="col">Confidence</th></tr>`;
      body.innerHTML = rows.length
        ? rows.map((row, i) => `
            <tr>
              <td>${offset + i + 1}</td>
              <td>${escapeHtml(row.actual != null ? row.actual : "—")}</td>
              <td>${escapeHtml(row.prediction != null ? row.prediction : "—")}</td>
              <td>${escapeHtml(formatConfidence(row.confidence))}</td>
            </tr>
          `).join("")
        : `<tr><td colspan="4" class="empty-inline">No prediction rows available.</td></tr>`;
    } else {
      head.innerHTML = `<tr><th scope="col">#</th><th scope="col">Actual</th><th scope="col">Prediction</th><th scope="col">Absolute Error</th></tr>`;
      body.innerHTML = rows.length
        ? rows.map((row, i) => `
            <tr>
              <td>${offset + i + 1}</td>
              <td>${escapeHtml(formatAdaptiveNumber(row.actual))}</td>
              <td>${escapeHtml(formatAdaptiveNumber(row.prediction))}</td>
              <td>${escapeHtml(formatAdaptiveNumber(row.absolute_error))}</td>
            </tr>
          `).join("")
        : `<tr><td colspan="4" class="empty-inline">No prediction rows available.</td></tr>`;
    }

    const from = total === 0 ? 0 : offset + 1;
    const to = Math.min(offset + limit, total);
    info.textContent = `Showing ${from}–${to} of ${total} rows`;
    prevBtn.disabled = offset <= 0;
    nextBtn.disabled = offset + limit >= total;
  }

  async function loadPredictionRowsPage(newOffset) {
    if (!currentRunId || !currentPredictionId || !currentPredictionResult) return;
    const prevBtn = byId("btn-pred-prev-page");
    const nextBtn = byId("btn-pred-next-page");
    prevBtn.disabled = true;
    nextBtn.disabled = true;

    try {
      const response = await LYNXApiClient.getPredictionRows(currentRunId, currentPredictionId, {
        offset: newOffset,
        limit: PREDICTION_ROWS_LIMIT,
      });
      const isClassification = currentPredictionResult.task_type === "classification";
      renderPredictionTable(isClassification, response);
    } catch (error) {
      prevBtn.disabled = predictionRowsOffset <= 0;
      nextBtn.disabled = predictionRowsOffset + PREDICTION_ROWS_LIMIT >= predictionRowsTotal;
    }
  }

  function renderInsights() {
    const context = ((currentReport.semantic_profile || {}).dataset_context) || {};
    if (context.dataset_description) byId("insights-context").textContent = `${context.dataset_description} No model-backed explanations are available yet.`;
  }

  function renderVisualizations() {
    const breakdown = currentDataset.issueStatusBreakdown.filter((item) => item.count > 0);
    const missing = (((currentReport.before_profile || {}).column_profiles) || [])
      .map((profile) => ({ name: profile.column_name, value: number((profile.missing || {}).percentage) }))
      .filter((item) => item.value > 0).sort((a, b) => b.value - a.value).slice(0, 8);
    const panels = [];
    if (breakdown.length) panels.push(chartPanel("Issue status", breakdown.map((item) => ({ name: item.label, value: item.count, display: item.count.toLocaleString() }))));
    if (missing.length) panels.push(chartPanel("Highest missingness", missing.map((item) => ({ ...item, display: `${item.value.toFixed(2)}%` }))));
    byId("visualization-grid").innerHTML = panels.length ? panels.join("") : `<p class="empty-inline">The report does not contain enough evidence for a supported visualization.</p>`;
  }

  function chartPanel(title, items) {
    const maximum = Math.max(...items.map((item) => item.value), 1);
    return `<section class="chart-panel"><h3>${escapeHtml(title)}</h3><div class="bar-list">${items.map((item) => `<div><div class="bar-label"><span title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span><strong>${escapeHtml(item.display)}</strong></div><div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, item.value / maximum * 100).toFixed(2)}%"></div></div></div>`).join("")}</div></section>`;
  }

  window.LYNX = {
    STATES,
    showState,
    activateTab,
    setTargetColumns,
    validateDatasetFile: fileValidation,
    beginPrediction,
    showPredictionSubView,
    renderPredictionResults,
    loadPredictionRowsPage,
    getCurrentState: () => currentState,
    getCurrentFile: () => selectedFile,
    getCurrentRunId: () => currentRunId,
    getCurrentRunStatus: () => currentRunStatus,
    getPipeline1Result: () => pipeline1Result,
    getProfileColumns: () => [...profileColumns],
    getCurrentPredictionId: () => currentPredictionId,
    getCurrentPredictionResult: () => currentPredictionResult,
    getPredictionController: () => predictionController,
  };
})();
