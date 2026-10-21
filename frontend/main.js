(function () {
  "use strict";

  const STATES = { UPLOAD: "upload", ANALYSIS: "analysis", WORKSPACE: "workspace", DATASET: "dataset", EXPORT: "export", SETTINGS: "settings" };
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
  let currentUploadStartedAt = null;
  let profileColumns = [];
  let selectedProfileColumn = null;
  let selectedProfileIndex = null;
  let profilingManifest = null;
  let profilePage = 1;
  const PROFILE_PAGE_SIZE = 10;
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
  let predictionResolutionToken = 0;
  let currentTaskResolution = null;
  let insightConversation = null;
  let insightsLanding = null;
  let insightsLandingDataset = null;
  let visualizationConversation = null;
  let automaticVisualizationDataset = null;
  let visualizationRenderer = null;
  let datasetRequestToken = 0;
  let datasetSearchTimer = null;
  let exportAvailability = null;
  const SETTINGS_DEFAULTS = { theme: "system", density: "comfortable", animations: true, confirmReplace: true, postUpload: "overview", showSuggestions: true };
  const settingKeys = { theme: "lynx-theme", density: "lynx-density", animations: "lynx-animations", confirmReplace: "lynx-confirm-replace", postUpload: "lynx-post-upload", showSuggestions: "lynx-show-suggestions" };
  let settingsPreferences = loadSettingsPreferences();
  let applicationInfo = null;
  const datasetExplorer = {
    version: "cleaned", offset: 0, limit: 20, search: "",
    sortColumn: "", sortDirection: "asc", filterColumn: "", filterValue: "",
    payload: null,
  };

  const byId = (id) => document.getElementById(id);
  const number = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;
  const finiteNumber = (value) => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value)) ? Number(value) : null;
  const escapeHtml = (value) => String(value == null ? "" : value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  const humanize = (value) => {
    const text = String(value || "").replace(/_/g, " ").trim();
    return text ? text.charAt(0).toUpperCase() + text.slice(1) : "Not available";
  };

  document.addEventListener("DOMContentLoaded", () => {
    visualizationRenderer = LYNXVisualizationController.createVisualizationController({
      adapter: LYNXEChartsAdapter,
      legacyRenderer: LYNXChartRenderer,
    });
    setupTheme();
    setupSettingsWorkspace();
    setupRunController();
    setupUpload();
    setupWorkspaceNavigation();
    setupGlobalActions();
    setupProfileControls();
    setupPredictionBoundary();
    setupConversationalSurfaces();
    setupDatasetExplorer();
    setupExportWorkspace();
    showState(STATES.UPLOAD);
  });

  function showState(state) {
    if (!Object.values(STATES).includes(state)) return;
    currentState = state;
    document.body.dataset.appState = state;
    document.querySelectorAll(".app-view").forEach((view) => {
      const active = view.id === `view-${state}`;
      view.classList.toggle("active", active);
      view.setAttribute("aria-hidden", String(!active));
    });
    const actions = byId("global-actions");
    const workspaceOnlyActions = [byId("btn-new-dataset"), byId("btn-export"), byId("more-new"), byId("more-export")];
    if (actions) actions.hidden = false;
    workspaceOnlyActions.forEach((action) => { if (action) action.hidden = ![STATES.WORKSPACE, STATES.DATASET, STATES.EXPORT].includes(state); });
    const moreMenu = byId("more-menu");
    const moreButton = byId("btn-more");
    if (moreMenu) moreMenu.hidden = true;
    if (moreButton) moreButton.setAttribute("aria-expanded", "false");
    const homeActive = state === STATES.UPLOAD;
    const analysisActive = state === STATES.ANALYSIS || state === STATES.WORKSPACE;
    const homeNav = byId("nav-home");
    const analysisNav = byId("nav-analysis");
    const datasetNav = byId("nav-datasets");
    const exportNav = byId("nav-export");
    const settingsNav = byId("nav-setting");
    if (homeNav) {
      homeNav.classList.toggle("active", homeActive);
      if (homeActive) homeNav.setAttribute("aria-current", "page");
      else homeNav.removeAttribute("aria-current");
    }
    if (analysisNav) {
      analysisNav.classList.toggle("active", analysisActive);
      analysisNav.setAttribute("aria-disabled", String(!currentReport && state !== STATES.ANALYSIS));
      analysisNav.title = !currentReport && state !== STATES.ANALYSIS ? "Upload and analyze a dataset first" : "Open dataset analysis";
      if (analysisActive) analysisNav.setAttribute("aria-current", "page");
      else analysisNav.removeAttribute("aria-current");
    }
    if (datasetNav) {
      const active = state === STATES.DATASET;
      datasetNav.disabled = !currentReport;
      datasetNav.title = currentReport ? "Explore original and cleaned dataset rows" : "Upload and analyze a dataset first";
      datasetNav.classList.toggle("active", active);
      if (active) datasetNav.setAttribute("aria-current", "page");
      else datasetNav.removeAttribute("aria-current");
    }
    if (exportNav) {
      const active = state === STATES.EXPORT;
      exportNav.disabled = !currentReport;
      exportNav.classList.toggle("sidebar-link-unavailable", !currentReport);
      exportNav.title = currentReport ? "Download prepared data and analysis reports" : "Export is available from completed results";
      exportNav.classList.toggle("active", active);
      if (active) exportNav.setAttribute("aria-current", "page");
      else exportNav.removeAttribute("aria-current");
    }
    if (settingsNav) {
      const active = state === STATES.SETTINGS;
      settingsNav.classList.toggle("active", active);
      if (active) settingsNav.setAttribute("aria-current", "page");
      else settingsNav.removeAttribute("aria-current");
    }
    [byId("btn-export"), byId("more-export")].forEach((button) => {
      if (!button) return;
      button.disabled = !currentReport;
      button.title = currentReport ? "Open Export Workspace" : "Export is available from completed results";
    });
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
    const choose = byId("btn-choose-file");
    const dropzone = byId("upload-dropzone");
    const input = byId("dataset-file-input");
    if (run) run.disabled = busy || !selectedFile;
    if (replace) replace.disabled = busy;
    if (choose) choose.disabled = busy;
    if (dropzone) dropzone.setAttribute("aria-busy", String(busy));
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
    currentUploadStartedAt = null;
    const recentList = byId("sidebar-recents-list");
    if (recentList) recentList.innerHTML = '<p class="sidebar-recents-empty">Your recent datasets will appear here.</p>';
    profileColumns = [];
    selectedProfileColumn = null;
    selectedProfileIndex = null;
    profilingManifest = null;
    profilePage = 1;
    targetColumns = [];
    predictionEligible = false;
    currentPredictionId = null;
    currentPredictionResult = null;
    currentPredictionTarget = null;
    predictionRowsOffset = 0;
    predictionRowsTotal = 0;
    automaticVisualizationDataset = null;
    datasetRequestToken += 1;
    datasetExplorer.version = "cleaned";
    datasetExplorer.offset = 0;
    datasetExplorer.limit = 20;
    datasetExplorer.search = "";
    datasetExplorer.sortColumn = "";
    datasetExplorer.sortDirection = "asc";
    datasetExplorer.filterColumn = "";
    datasetExplorer.filterValue = "";
    datasetExplorer.payload = null;
    exportAvailability = null;
    if (byId("dataset-search")) byId("dataset-search").value = "";
    if (byId("dataset-filter-value")) byId("dataset-filter-value").value = "";
    if (byId("dataset-page-size")) byId("dataset-page-size").value = "20";
    if (byId("dataset-grid-head")) byId("dataset-grid-head").innerHTML = "";
    if (byId("dataset-grid-body")) byId("dataset-grid-body").innerHTML = "";
    if (byId("dataset-row-range")) byId("dataset-row-range").textContent = "Rows 0–0 of 0";
    if (byId("dataset-change-toggle")) updateDatasetVersionControls();
    insightsLanding = null;
    insightsLandingDataset = null;
    if (insightConversation) insightConversation.reset();
    if (visualizationConversation) visualizationConversation.reset();
    clearAutomaticVisualizations();
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
    const choose = byId("btn-choose-file");
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
    dropzone.addEventListener("click", (event) => { if (!event.target.closest("button")) input.click(); });
    choose.addEventListener("click", () => input.click());
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
        showCompletedAnalysis(response);
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
    currentUploadStartedAt = new Date();
    controller.start(selectedFile);
  }

  function showCompletedAnalysis(response) {
    if (!response || !response.pipeline1_report || !response.prepared_dataset) {
      throw new TypeError("A completed analysis response is required.");
    }
    stopElapsed();
    setUploadBusy(false);
    pipeline1Result = response;
    currentReport = response.pipeline1_report;
    currentRunId = response.run_id;
    currentRunStatus = { ...(currentRunStatus || {}), ...response };
    const reportLocation = `/api/runs/${encodeURIComponent(response.run_id)}/pipeline1`;
    currentDataset = LYNXReportAdapter.adaptCleaningReport(response.pipeline1_report, { reportLocation });
    if (response.profiling_manifest && response.profiling_manifest.dataset_name) {
      currentDataset.name = String(response.profiling_manifest.dataset_name);
    }
    if (insightConversation) insightConversation.setDataset(response.run_id);
    if (visualizationConversation) visualizationConversation.setDataset(response.run_id);
    renderWorkspace();
    setTargetColumns(response.prepared_dataset.target_columns);
    if (settingsPreferences.postUpload === "dataset") {
      showState(STATES.DATASET);
      loadDatasetPage();
    } else {
      showState(STATES.WORKSPACE);
      activateTab("overview");
    }
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
    if (tab === "visualizations" && visualizationRenderer) {
      requestAnimationFrame(() => visualizationRenderer.resizeWithin(byId("panel-visualizations")));
    }
  }

  function setupGlobalActions() {
    byId("btn-new-dataset").addEventListener("click", requestNewDataset);
    byId("brand-home").addEventListener("click", () => showState(STATES.UPLOAD));
    byId("mobile-brand-home").addEventListener("click", () => showState(STATES.UPLOAD));
    byId("nav-home").addEventListener("click", () => showState(STATES.UPLOAD));
    byId("nav-analysis").addEventListener("click", () => {
      if (currentState === STATES.ANALYSIS) return;
      if (currentReport) showState(STATES.WORKSPACE);
    });
    byId("nav-datasets").addEventListener("click", () => {
      if (!currentReport) return;
      showState(STATES.DATASET);
      loadDatasetPage();
    });
    const openExport = () => {
      if (!currentReport) return;
      showState(STATES.EXPORT);
      loadExportAvailability();
    };
    byId("nav-export").addEventListener("click", openExport);
    byId("btn-export").addEventListener("click", openExport);
    const openSettings = () => { showState(STATES.SETTINGS); loadApplicationInfo(); };
    byId("btn-settings").addEventListener("click", openSettings);
    byId("nav-setting").addEventListener("click", openSettings);
    byId("btn-help").addEventListener("click", () => byId("help-dialog").showModal());
    document.querySelectorAll("[data-shell-action]").forEach((button) => {
      if (button.dataset.shellAction !== "settings") button.addEventListener("click", () => byId(`${button.dataset.shellAction}-dialog`).showModal());
    });
    const moreButton = byId("btn-more");
    const menu = byId("more-menu");
    moreButton.addEventListener("click", () => {
      menu.hidden = !menu.hidden;
      moreButton.setAttribute("aria-expanded", String(!menu.hidden));
    });
    menu.addEventListener("click", (event) => {
      const action = event.target.dataset.moreAction;
      if (action === "settings") openSettings();
      if (action === "help") byId("help-dialog").showModal();
      if (action === "new") requestNewDataset();
      if (action === "export") openExport();
      if (action === "theme") toggleTheme();
      menu.hidden = true;
      moreButton.setAttribute("aria-expanded", "false");
    });
    document.addEventListener("click", (event) => {
      if (!event.target.closest(".more-wrap")) { menu.hidden = true; moreButton.setAttribute("aria-expanded", "false"); }
    });
  }

  function requestNewDataset() {
    if (settingsPreferences.confirmReplace && currentReport && !window.confirm("Replace the active dataset and clear this browser workspace?")) return;
    startNewDataset();
  }

  function currentTheme() {
    return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  }

  function applyTheme(theme, persist = true) {
    const mode = ["light", "dark", "system"].includes(theme) ? theme : "system";
    const next = mode === "system" ? (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light") : mode;
    document.documentElement.dataset.theme = next;
    document.documentElement.dataset.themeMode = mode;
    settingsPreferences.theme = mode;
    if (persist) {
      saveSetting("theme", mode);
    }
    const toggle = byId("btn-theme");
    const label = byId("theme-label");
    if (toggle) {
      toggle.setAttribute("aria-pressed", String(next === "dark"));
      toggle.setAttribute("aria-label", `Switch to ${next === "dark" ? "light" : "dark"} theme`);
    }
    if (label) label.textContent = next === "dark" ? "Light" : "Dark";
    document.querySelectorAll('input[name="settings-theme"]').forEach((choice) => { choice.checked = choice.value === mode; });
  }

  function toggleTheme() { applyTheme(currentTheme() === "dark" ? "light" : "dark"); }

  function setupTheme() {
    applyTheme(settingsPreferences.theme, false);
    byId("btn-theme").addEventListener("click", toggleTheme);
    if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (settingsPreferences.theme === "system") applyTheme("system", false); });
  }

  function storedSetting(name, fallback) {
    try {
      const value = window.localStorage.getItem(settingKeys[name]);
      if (value === null) return fallback;
      if (typeof fallback === "boolean") return value !== "false";
      return value;
    } catch (_error) { return fallback; }
  }

  function loadSettingsPreferences() {
    const preferences = {
      theme: storedSetting("theme", SETTINGS_DEFAULTS.theme),
      density: storedSetting("density", SETTINGS_DEFAULTS.density),
      animations: storedSetting("animations", SETTINGS_DEFAULTS.animations),
      confirmReplace: storedSetting("confirmReplace", SETTINGS_DEFAULTS.confirmReplace),
      postUpload: storedSetting("postUpload", SETTINGS_DEFAULTS.postUpload),
      showSuggestions: storedSetting("showSuggestions", SETTINGS_DEFAULTS.showSuggestions),
    };
    if (!["light", "dark", "system"].includes(preferences.theme)) preferences.theme = SETTINGS_DEFAULTS.theme;
    if (!["comfortable", "compact"].includes(preferences.density)) preferences.density = SETTINGS_DEFAULTS.density;
    if (!["overview", "dataset"].includes(preferences.postUpload)) preferences.postUpload = SETTINGS_DEFAULTS.postUpload;
    return preferences;
  }

  function saveSetting(name, value) {
    settingsPreferences[name] = value;
    try {
      window.localStorage.setItem(settingKeys[name], String(value));
      const status = byId("settings-save-status");
      if (status) status.textContent = "Preference saved.";
    } catch (_error) {
      const status = byId("settings-save-status");
      if (status) status.textContent = "This preference could not be saved in this browser.";
    }
  }

  function applySettingsPreferences() {
    document.documentElement.dataset.density = settingsPreferences.density;
    document.documentElement.dataset.animations = settingsPreferences.animations ? "on" : "off";
    document.querySelectorAll('input[name="settings-density"]').forEach((input) => { input.checked = input.value === settingsPreferences.density; });
    byId("setting-animations").checked = settingsPreferences.animations;
    byId("setting-confirm-replace").checked = settingsPreferences.confirmReplace;
    byId("setting-post-upload").value = settingsPreferences.postUpload;
    byId("setting-suggestions").checked = settingsPreferences.showSuggestions;
    applySuggestionVisibility();
  }

  function applySuggestionVisibility() {
    const card = document.querySelector(".insights-suggestions");
    if (card) card.hidden = !settingsPreferences.showSuggestions;
  }

  function showSettingsCategory(category) {
    document.querySelectorAll("[data-settings-category]").forEach((button) => {
      const active = button.dataset.settingsCategory === category;
      button.classList.toggle("active", active);
      if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
    });
    document.querySelectorAll("[data-settings-panel]").forEach((panel) => {
      const active = panel.dataset.settingsPanel === category;
      panel.classList.toggle("active", active);
      panel.hidden = !active;
    });
  }

  function setupSettingsWorkspace() {
    applySettingsPreferences();
    document.querySelectorAll("[data-settings-category]").forEach((button) => button.addEventListener("click", () => showSettingsCategory(button.dataset.settingsCategory)));
    document.querySelectorAll('input[name="settings-theme"]').forEach((input) => input.addEventListener("change", () => applyTheme(input.value)));
    document.querySelectorAll('input[name="settings-density"]').forEach((input) => input.addEventListener("change", () => {
      document.documentElement.dataset.density = input.value;
      saveSetting("density", input.value);
    }));
    byId("setting-animations").addEventListener("change", (event) => {
      document.documentElement.dataset.animations = event.target.checked ? "on" : "off";
      saveSetting("animations", event.target.checked);
    });
    byId("setting-confirm-replace").addEventListener("change", (event) => saveSetting("confirmReplace", event.target.checked));
    byId("setting-post-upload").addEventListener("change", (event) => saveSetting("postUpload", event.target.value));
    byId("setting-suggestions").addEventListener("change", (event) => { saveSetting("showSuggestions", event.target.checked); applySuggestionVisibility(); });
    byId("btn-reset-appearance").addEventListener("click", () => {
      applyTheme(SETTINGS_DEFAULTS.theme);
      saveSetting("density", SETTINGS_DEFAULTS.density);
      saveSetting("animations", SETTINGS_DEFAULTS.animations);
      applySettingsPreferences();
      byId("settings-save-status").textContent = "Appearance preferences reset.";
    });
    byId("btn-clear-workspace").addEventListener("click", () => {
      if (!currentReport) { byId("settings-save-status").textContent = "There is no active browser workspace to clear."; return; }
      if (!window.confirm("Clear the active browser workspace? Server artifacts remain subject to the configured retention policy.")) return;
      resetAnalysisState();
      selectedFile = null;
      const input = byId("dataset-file-input");
      if (input) input.value = "";
      byId("selected-file-card").hidden = true;
      showState(STATES.SETTINGS);
      byId("settings-save-status").textContent = "Active browser workspace cleared.";
    });
  }

  function formatRetention(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value) || value <= 0) return "Not reported";
    if (value % 86400 === 0) return `${value / 86400} day${value === 86400 ? "" : "s"}`;
    if (value % 3600 === 0) return `${value / 3600} hour${value === 3600 ? "" : "s"}`;
    return `${value} seconds`;
  }

  async function loadApplicationInfo() {
    if (applicationInfo) return;
    try {
      applicationInfo = await LYNXApiClient.getApplicationInfo();
      const ai = applicationInfo.ai || {};
      const data = applicationInfo.data || {};
      byId("settings-ai-provider").textContent = ai.provider || "Not configured";
      byId("settings-ai-model").textContent = ai.model || "Not reported";
      byId("settings-ai-status").textContent = ai.status === "configured" ? "Configured" : "Not configured";
      byId("settings-ai-status").classList.toggle("required", ai.status === "configured");
      byId("settings-data-storage").textContent = data.storage || "Runtime-managed storage";
      byId("settings-retention").textContent = formatRetention(data.retention_seconds);
      byId("settings-app-version").textContent = applicationInfo.application && applicationInfo.application.version ? applicationInfo.application.version : "Not reported";
    } catch (_error) {
      ["settings-ai-provider", "settings-ai-model", "settings-retention", "settings-app-version"].forEach((id) => { byId(id).textContent = "Temporarily unavailable"; });
      byId("settings-ai-status").textContent = "Unavailable";
      byId("settings-save-status").textContent = "Application information could not be loaded. Preferences remain available.";
    }
  }

  function renderWorkspace() {
    const rows = number(pipeline1Result.prepared_dataset.rows);
    const columns = number(pipeline1Result.prepared_dataset.columns);
    byId("workspace-dataset-name").textContent = currentDataset.name;
    byId("workspace-dataset-name").title = currentDataset.name;
    byId("prediction-dataset-name").textContent = currentDataset.name;
    byId("insights-dataset-name").textContent = currentDataset.name;
    byId("insights-dataset-name").title = currentDataset.name;
    byId("visualization-dataset-name").textContent = currentDataset.name;
    byId("visualization-dataset-name").title = currentDataset.name;
    byId("dataset-page-name").textContent = currentDataset.name;
    byId("dataset-page-name").title = currentDataset.name;
    const extension = String(currentDataset.name || "").split(".").pop();
    byId("workspace-file-type").textContent = extension && extension !== currentDataset.name ? extension.toUpperCase() : "DATA";
    byId("workspace-dataset-meta").textContent = "Uploaded dataset overview";
    renderSidebarCurrentDataset();
    renderOverview();
    initializeDataProfile();
    loadInsightsLanding();
    loadAutomaticVisualizations();
  }

  function renderOverview() {
    const report = currentReport;
    const metrics = currentDataset.metrics;
    const validation = report.validation || {};
    byId("overview-summary").textContent = currentDataset.summary || "Analysis completed using the evidence available in the dataset.";
    const after = validation.after_summary || {};
    const datasetOverview = pipeline1Result.dataset_overview || {};
    const totalMissing = finiteNumber(after.total_missing_cells);
    const cellCount = metrics.rows * metrics.columns;
    const missingPercentage = totalMissing !== null && cellCount > 0 ? (totalMissing / cellCount) * 100 : null;
    const completeness = missingPercentage === null ? null : Math.max(0, 100 - missingPercentage);
    const uploadTimestamp = (currentRunStatus && (currentRunStatus.created_at || currentRunStatus.submitted_at || currentRunStatus.started_at)) || currentUploadStartedAt;
    byId("overview-uploaded-at").querySelector("span").textContent = uploadTimestamp
      ? `Uploaded on ${formatOverviewDate(uploadTimestamp)}`
      : "Upload time unavailable";

    const memoryBytes = finiteNumber(datasetOverview.memory_footprint_bytes);
    const kpis = [
      { tone: "purple", icon: "database", label: "Total Rows", value: metrics.rows.toLocaleString(), help: "Rows in dataset" },
      { tone: "blue", icon: "table", label: "Total Columns", value: metrics.columns.toLocaleString(), help: "Columns in dataset" },
      { tone: "green", icon: "file", label: "Memory Footprint", value: memoryBytes === null ? "Unavailable" : formatBytes(memoryBytes), help: memoryBytes === null ? "Measurement unavailable" : "Original ingestion DataFrame" },
      { tone: "orange", icon: "percent", label: "Missing Value Percentage", value: missingPercentage === null ? "Unavailable" : `${missingPercentage.toFixed(1)}%`, help: "Cells with missing values" },
    ];
    byId("overview-metrics").innerHTML = kpis.map((item) => `<article class="overview-kpi"><span class="kpi-icon ${item.tone}">${overviewIcon(item.icon)}</span><div><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong><small>${escapeHtml(item.help)}</small></div></article>`).join("");

    const completionText = completeness === null ? "—" : `${completeness.toFixed(1)}%`;
    const missingText = missingPercentage === null ? "—" : `${missingPercentage.toFixed(1)}%`;
    const issueText = `${number(metrics.issuesCount).toLocaleString()} recorded`;
    const donutValue = completeness === null ? 0 : completeness;
    byId("overview-quality").innerHTML = `<div class="quality-overview"><div class="quality-ring${completeness === null ? " unavailable" : ""}" style="--quality-value:${donutValue.toFixed(2)}"><div><strong>${escapeHtml(completionText)}</strong><span>Observed completeness</span></div></div><dl class="quality-legend"><div><dt><span class="legend-dot valid"></span>Complete cells</dt><dd>${escapeHtml(completionText)}</dd></div><div><dt><span class="legend-dot missing"></span>Missing cells</dt><dd>${escapeHtml(missingText)}</dd></div><div><dt><span class="legend-dot findings"></span>Quality findings</dt><dd>${escapeHtml(issueText)}</dd></div></dl></div>`;

    const passed = validation.passed === true;
    const privacyPassed = pipeline1Result.privacy_clearance_passed === true;
    const checks = [
      [validation.row_count_equal, "Row count preserved"],
      [validation.column_count_equal, "Column count preserved"],
      [validation.only_approved_value_changes, "Only approved value changes"],
      [validation.missingness_equal || validation.expected_missingness_changes_only, "Missingness checks completed"],
      [privacyPassed, "Privacy clearance passed"],
      [passed && report.status === "success", "Ready for analysis"],
    ];
    byId("overview-health").innerHTML = `<div class="health-summary"><span class="health-icon ${passed ? "pass" : "review"}" aria-hidden="true">${passed ? "✓" : "!"}</span><div><div class="health-title-line"><strong>${passed ? "Validation Passed" : "Review Required"}</strong><span class="health-badge ${passed ? "pass" : "review"}">${passed ? "Pass" : "Review"}</span></div><p>${passed ? "The prepared dataset passed the available validation checks." : "One or more validation checks require review."}</p></div></div><ul class="health-checks">${checks.map(([ok, label]) => `<li class="${ok ? "pass" : "unavailable"}"><span aria-hidden="true">${ok ? "✓" : "—"}</span>${escapeHtml(label)}</li>`).join("")}</ul>`;

    renderRawDataPreview(datasetOverview.raw_data_preview);
  }

  function renderRawDataPreview(preview) {
    const target = byId("overview-raw-preview");
    const description = byId("overview-preview-description");
    if (!preview || preview.available !== true) {
      const privacyBlocked = preview && preview.unavailable_reason === "privacy_clearance_not_passed";
      description.textContent = privacyBlocked
        ? "Preview is withheld because privacy clearance did not pass."
        : "Original row preview is unavailable for this analysis.";
      target.innerHTML = `<div class="preview-unavailable"><strong>Preview unavailable</strong><span>${privacyBlocked ? "Raw values are never exposed without privacy clearance." : "The original ingestion data could not be safely prepared for display."}</span></div>`;
      return;
    }
    const columns = Array.isArray(preview.columns) ? preview.columns : [];
    const rows = Array.isArray(preview.rows) ? preview.rows : [];
    const redacted = Array.isArray(preview.privacy_redacted_columns) ? preview.privacy_redacted_columns : [];
    const boundedNote = preview.columns_truncated ? ` The preview is limited to the first ${columns.length} columns.` : "";
    const privacyNote = redacted.length ? ` Privacy-protected values are redacted in ${redacted.length} column${redacted.length === 1 ? "" : "s"}.` : "";
    description.textContent = `Showing ${rows.length} of ${number(preview.total_row_count).toLocaleString()} original rows.${boundedNote}${privacyNote}`;
    if (!columns.length) {
      target.innerHTML = `<div class="preview-unavailable"><strong>Dataset is empty</strong><span>No columns are available to preview.</span></div>`;
      return;
    }
    const body = rows.length
      ? rows.map((row, rowIndex) => `<tr><td>${rowIndex + 1}</td>${columns.map((_column, columnIndex) => previewCellMarkup(Array.isArray(row) ? row[columnIndex] : null)).join("")}</tr>`).join("")
      : `<tr><td colspan="${columns.length + 1}"><div class="preview-unavailable"><strong>No data rows</strong><span>The ingested dataset contains columns but no rows.</span></div></td></tr>`;
    target.innerHTML = `<div class="preview-scroll"><table><caption class="sr-only">First ${rows.length} original dataset rows</caption><thead><tr><th>#</th>${columns.map((name) => `<th>${escapeHtml(name)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  function previewCellMarkup(value) {
    if (value === null || value === undefined) return '<td class="preview-missing" aria-label="Missing value">—</td>';
    const text = typeof value === "object" ? JSON.stringify(value) : String(value);
    const protectedValue = text === "[REDACTED]";
    return `<td${protectedValue ? ' class="preview-redacted"' : ""} title="${escapeHtml(text)}">${escapeHtml(text)}</td>`;
  }

  function formatOverviewDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "time unavailable";
    return date.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
  }

  function overviewIcon(name) {
    const icons = {
      database: '<svg aria-hidden="true" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v6c0 1.7 3.1 3 7 3s7-1.3 7-3V5M5 11v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/></svg>',
      table: '<svg aria-hidden="true" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 3v18M15 3v18M3 15h18"/></svg>',
      file: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M6 3h8l4 4v14H6zM14 3v5h4M9 13h6M9 17h6"/></svg>',
      percent: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="m5 19 14-14"/><circle cx="7" cy="7" r="2.5"/><circle cx="17" cy="17" r="2.5"/></svg>',
    };
    return icons[name] || "";
  }

  function renderSidebarCurrentDataset() {
    const list = byId("sidebar-recents-list");
    if (!list || !currentDataset) return;
    list.innerHTML = `<button class="sidebar-recent-item active" type="button" aria-current="page" title="${escapeHtml(currentDataset.name)}"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="M6 3h8l4 4v14H6zM14 3v5h4"/></svg><span>${escapeHtml(currentDataset.name)}</span></button>`;
    list.querySelector("button").addEventListener("click", () => { showState(STATES.WORKSPACE); activateTab("overview"); });
  }

  function setupDatasetExplorer() {
    document.querySelectorAll("[data-dataset-version]").forEach((button) => {
      button.addEventListener("click", () => {
        const version = button.dataset.datasetVersion;
        if (version === datasetExplorer.version) return;
        datasetExplorer.version = version;
        datasetExplorer.offset = 0;
        datasetExplorer.sortColumn = "";
        datasetExplorer.sortDirection = "asc";
        datasetExplorer.filterColumn = "";
        datasetExplorer.filterValue = "";
        byId("dataset-filter-value").value = "";
        updateDatasetVersionControls();
        loadDatasetPage();
      });
    });
    byId("dataset-search").addEventListener("input", (event) => {
      datasetExplorer.search = event.target.value;
      datasetExplorer.offset = 0;
      queueDatasetLoad();
    });
    byId("dataset-filter-column").addEventListener("change", (event) => {
      datasetExplorer.filterColumn = event.target.value;
      datasetExplorer.offset = 0;
      const input = byId("dataset-filter-value");
      input.disabled = !datasetExplorer.filterColumn;
      if (!datasetExplorer.filterColumn) {
        input.value = "";
        datasetExplorer.filterValue = "";
      }
      loadDatasetPage();
    });
    byId("dataset-filter-value").addEventListener("input", (event) => {
      datasetExplorer.filterValue = event.target.value;
      datasetExplorer.offset = 0;
      queueDatasetLoad();
    });
    byId("dataset-sort-column").addEventListener("change", (event) => {
      datasetExplorer.sortColumn = event.target.value;
      datasetExplorer.offset = 0;
      byId("dataset-sort-direction").disabled = !datasetExplorer.sortColumn;
      loadDatasetPage();
    });
    byId("dataset-sort-direction").addEventListener("click", () => {
      datasetExplorer.sortDirection = datasetExplorer.sortDirection === "asc" ? "desc" : "asc";
      datasetExplorer.offset = 0;
      updateDatasetSortDirection();
      loadDatasetPage();
    });
    byId("dataset-page-size").addEventListener("change", (event) => {
      datasetExplorer.limit = Number(event.target.value);
      datasetExplorer.offset = 0;
      loadDatasetPage();
    });
    byId("dataset-prev-page").addEventListener("click", () => {
      datasetExplorer.offset = Math.max(0, datasetExplorer.offset - datasetExplorer.limit);
      loadDatasetPage();
    });
    byId("dataset-next-page").addEventListener("click", () => {
      const total = datasetExplorer.payload ? number(datasetExplorer.payload.filtered_rows) : 0;
      if (datasetExplorer.offset + datasetExplorer.limit < total) {
        datasetExplorer.offset += datasetExplorer.limit;
        loadDatasetPage();
      }
    });
    byId("dataset-page-buttons").addEventListener("click", (event) => {
      const button = event.target.closest("[data-dataset-page]");
      if (!button) return;
      datasetExplorer.offset = (Number(button.dataset.datasetPage) - 1) * datasetExplorer.limit;
      loadDatasetPage();
    });
    byId("dataset-grid-head").addEventListener("click", (event) => {
      const button = event.target.closest("[data-sort-dataset-column]");
      if (!button) return;
      if (datasetExplorer.sortColumn === button.dataset.sortDatasetColumn) {
        datasetExplorer.sortDirection = datasetExplorer.sortDirection === "asc" ? "desc" : "asc";
      } else {
        datasetExplorer.sortColumn = button.dataset.sortDatasetColumn;
        datasetExplorer.sortDirection = "asc";
      }
      datasetExplorer.offset = 0;
      updateDatasetSortDirection();
      loadDatasetPage();
    });
    updateDatasetVersionControls();
  }

  function queueDatasetLoad() {
    if (datasetSearchTimer !== null) window.clearTimeout(datasetSearchTimer);
    datasetSearchTimer = window.setTimeout(loadDatasetPage, 250);
  }

  function updateDatasetVersionControls() {
    document.querySelectorAll("[data-dataset-version]").forEach((button) => {
      const active = button.dataset.datasetVersion === datasetExplorer.version;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    const toggle = byId("dataset-change-toggle");
    toggle.hidden = datasetExplorer.version === "original";
    byId("dataset-highlight-changes").checked = false;
  }

  function updateDatasetSortDirection() {
    const button = byId("dataset-sort-direction");
    const descending = datasetExplorer.sortDirection === "desc";
    button.textContent = descending ? "↓" : "↑";
    button.setAttribute("aria-label", descending ? "Sort descending" : "Sort ascending");
    button.setAttribute("aria-pressed", String(descending));
  }

  async function loadDatasetPage() {
    if (!currentRunId) return;
    const token = ++datasetRequestToken;
    const state = byId("dataset-state");
    state.className = "dataset-state loading";
    state.textContent = `Loading ${datasetExplorer.version} dataset…`;
    try {
      const payload = await LYNXApiClient.getDatasetPage(currentRunId, {
        version: datasetExplorer.version,
        offset: datasetExplorer.offset,
        limit: datasetExplorer.limit,
        search: datasetExplorer.search,
        sort_column: datasetExplorer.sortColumn,
        sort_direction: datasetExplorer.sortDirection,
        filter_column: datasetExplorer.filterColumn,
        filter_value: datasetExplorer.filterValue,
      });
      if (token !== datasetRequestToken || currentState !== STATES.DATASET) return;
      datasetExplorer.payload = payload;
      renderDatasetPage(payload);
    } catch (error) {
      if (token !== datasetRequestToken || currentState !== STATES.DATASET) return;
      datasetExplorer.payload = null;
      byId("dataset-grid-head").innerHTML = "";
      byId("dataset-grid-body").innerHTML = "";
      state.className = "dataset-state error";
      state.textContent = error && error.message ? error.message : "Dataset rows could not be retrieved.";
      renderDatasetPagination(0);
    }
  }

  function renderDatasetPage(payload) {
    const columns = Array.isArray(payload.columns) ? payload.columns : [];
    const rows = Array.isArray(payload.rows) ? payload.rows : [];
    const rowNumbers = Array.isArray(payload.row_numbers) ? payload.row_numbers : [];
    populateDatasetColumnSelect(byId("dataset-filter-column"), "Filter column", columns, datasetExplorer.filterColumn);
    populateDatasetColumnSelect(byId("dataset-sort-column"), "Sort column", columns, datasetExplorer.sortColumn);
    byId("dataset-filter-value").disabled = !datasetExplorer.filterColumn;
    byId("dataset-sort-direction").disabled = !datasetExplorer.sortColumn;
    updateDatasetSortDirection();
    const state = byId("dataset-state");
    const empty = rows.length === 0;
    state.className = "dataset-state";
    state.textContent = empty
      ? ((datasetExplorer.search || datasetExplorer.filterValue) ? "No rows match the current search or filter." : "This dataset version contains no rows.")
      : "";
    const sortIndicator = (column) => datasetExplorer.sortColumn === column
      ? `<span aria-hidden="true">${datasetExplorer.sortDirection === "desc" ? "↓" : "↑"}</span>` : '<span aria-hidden="true">↕</span>';
    byId("dataset-grid-head").innerHTML = columns.length
      ? `<tr><th scope="col" class="dataset-row-number">#</th>${columns.map((column) => `<th scope="col"><button type="button" data-sort-dataset-column="${escapeHtml(column)}">${escapeHtml(column)}${sortIndicator(column)}</button></th>`).join("")}</tr>` : "";
    byId("dataset-grid-body").innerHTML = rows.map((row, rowIndex) => `<tr><th scope="row" class="dataset-row-number">${escapeHtml(rowNumbers[rowIndex] || datasetExplorer.offset + rowIndex + 1)}</th>${columns.map((_column, columnIndex) => datasetCellMarkup(row[columnIndex])).join("")}</tr>`).join("");
    renderDatasetPagination(number(payload.filtered_rows));
    const changeToggle = byId("dataset-change-toggle");
    changeToggle.title = payload.change_highlighting_reason || "Highlight verified cleaning changes";
    byId("dataset-highlight-changes").disabled = !payload.change_highlighting_available;
  }

  function datasetCellMarkup(value) {
    if (value === null || value === undefined || value === "") return '<td class="dataset-missing"><span aria-label="Missing value">—</span></td>';
    const display = typeof value === "object" ? JSON.stringify(value) : String(value);
    return `<td title="${escapeHtml(display)}">${escapeHtml(display)}</td>`;
  }

  function populateDatasetColumnSelect(select, label, columns, selected) {
    select.innerHTML = `<option value="">${label}</option>${columns.map((column) => `<option value="${escapeHtml(column)}">${escapeHtml(column)}</option>`).join("")}`;
    select.value = columns.includes(selected) ? selected : "";
    if (!select.value && selected) {
      if (select.id === "dataset-filter-column") datasetExplorer.filterColumn = "";
      if (select.id === "dataset-sort-column") datasetExplorer.sortColumn = "";
    }
  }

  function renderDatasetPagination(total) {
    const from = total ? datasetExplorer.offset + 1 : 0;
    const to = Math.min(datasetExplorer.offset + datasetExplorer.limit, total);
    byId("dataset-row-range").textContent = `Rows ${from.toLocaleString()}–${to.toLocaleString()} of ${total.toLocaleString()}`;
    const pageCount = Math.max(1, Math.ceil(total / datasetExplorer.limit));
    const currentPage = Math.min(pageCount, Math.floor(datasetExplorer.offset / datasetExplorer.limit) + 1);
    const start = Math.max(1, Math.min(currentPage - 2, pageCount - 4));
    const end = Math.min(pageCount, start + 4);
    const pages = [];
    for (let page = start; page <= end; page += 1) pages.push(page);
    byId("dataset-page-buttons").innerHTML = pages.map((page) => `<button type="button" data-dataset-page="${page}"${page === currentPage ? ' class="active" aria-current="page"' : ""}>${page}</button>`).join("");
    byId("dataset-prev-page").disabled = currentPage <= 1;
    byId("dataset-next-page").disabled = currentPage >= pageCount || total === 0;
  }

  function isResolvedSemantic(value) {
    return value != null && !["", "unknown", "unresolved", "not_available"].includes(String(value).trim().toLocaleLowerCase());
  }

  function semanticLabel(value) {
    return isResolvedSemantic(value) ? humanize(value) : "Unknown";
  }

  function actionLabel(actionId) {
    const labels = {
      no_op: "No action required", trim_whitespace: "Trim whitespace",
      normalize_categorical_labels: "Normalize categorical labels", parse_dates: "Parse dates",
      coerce_dtype: "Coerce data type", impute_missing: "Impute missing values",
      mask_column: "Mask for privacy", add_missing_indicator: "Add missing indicator",
      drop_duplicate_rows: "Remove duplicate rows", flag_rows: "Flag for review",
    };
    return labels[actionId] || humanize(actionId || "No action required");
  }

  function profileStatusLabel(status) {
    const labels = {
      completed: "Completed", partial: "Partial", unavailable: "Unavailable", failed: "Failed",
      proposed: "Proposed", applied: "Applied", skipped: "Skipped", not_required: "Not Required",
      requires_review: "Requires Review",
    };
    return labels[status] || humanize(status);
  }

  function legacyProfilingManifest(report) {
    const profiles = Array.isArray((report.before_profile || {}).column_profiles) ? report.before_profile.column_profiles : [];
    const semantics = Array.isArray((report.semantic_profile || {}).columns) ? report.semantic_profile.columns : [];
    const semanticByName = new Map(semantics.map((item) => [String(item.column_name), item]));
    return {
      dataset_name: currentDataset ? currentDataset.name : "Dataset",
      profiled_at: null,
      profile_status: profiles.length ? "partial" : "unavailable",
      total_columns: profiles.length,
      semantic_roles_detected: [],
      semantic_role_count: 0,
      columns_requiring_cleaning: 0,
      columns: profiles.map((profile, columnIndex) => {
        const semantic = semanticByName.get(String(profile.column_name)) || {};
        const missing = profile.missing || {};
        const unique = profile.uniqueness || {};
        return {
          column_index: columnIndex, column_name: String(profile.column_name),
          physical_dtype: profile.physical_dtype || null, semantic_role: semantic.semantic_role || null,
          semantic_type: semantic.semantic_type || null, meaning: semantic.concept_description || null,
          semantic_confidence: finiteNumber(semantic.confidence), non_null_count: number(profile.non_null_count),
          missing_count: number(missing.count), missing_percentage: number(missing.percentage),
          unique_count: number(unique.unique_count), primary_action: null, actions: [], status: "unavailable",
          decision_explanation: semantic.reasoning_summary || semantic.concept_description || null,
          explanation_source: semantic.reasoning_summary ? "semantic_reasoning" : semantic.concept_description ? "semantic_description" : "unavailable",
          observed_examples: [], privacy_restricted: true,
        };
      }),
    };
  }

  function initializeDataProfile() {
    profilingManifest = pipeline1Result.profiling_manifest || legacyProfilingManifest(currentReport);
    profileColumns = Array.isArray(profilingManifest.columns) ? profilingManifest.columns : [];
    selectedProfileIndex = profileColumns.length ? number(profileColumns[0].column_index) : null;
    selectedProfileColumn = profileColumns.length ? profileColumns[0].column_name : null;
    profilePage = 1;
    byId("column-search").value = "";
    const roleFilter = byId("semantic-role-filter");
    const roleValues = [...new Set(profileColumns.map((column) => isResolvedSemantic(column.semantic_role) ? String(column.semantic_role) : "__unknown__"))];
    roleFilter.innerHTML = `<option value="">All roles</option>${roleValues.sort((a, b) => semanticLabel(a).localeCompare(semanticLabel(b))).map((role) => `<option value="${escapeHtml(role)}">${escapeHtml(role === "__unknown__" ? "Unknown" : semanticLabel(role))}</option>`).join("")}`;
    roleFilter.value = "";
    renderProfilingSummary();
    renderProfilingWorkspace();
  }

  function setupProfileControls() {
    byId("column-search").addEventListener("input", () => { profilePage = 1; renderProfilingWorkspace(); });
    byId("semantic-role-filter").addEventListener("change", () => { profilePage = 1; renderProfilingWorkspace(); });
    byId("semantic-manifest-body").addEventListener("click", (event) => {
      const row = event.target.closest("tr[data-column-index]");
      if (row) selectProfileColumn(Number(row.dataset.columnIndex));
    });
    byId("semantic-manifest-body").addEventListener("keydown", (event) => {
      const row = event.target.closest("tr[data-column-index]");
      if (row && ["Enter", " "].includes(event.key)) {
        event.preventDefault();
        selectProfileColumn(Number(row.dataset.columnIndex));
      }
    });
    byId("manifest-page-prev").addEventListener("click", () => { if (profilePage > 1) { profilePage -= 1; renderProfilingWorkspace(); } });
    byId("manifest-page-next").addEventListener("click", () => {
      const pages = Math.max(1, Math.ceil(filteredProfileColumns().length / PROFILE_PAGE_SIZE));
      if (profilePage < pages) { profilePage += 1; renderProfilingWorkspace(); }
    });
    byId("manifest-page-buttons").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-page]");
      if (button) { profilePage = Number(button.dataset.page); renderProfilingWorkspace(); }
    });
  }

  function filteredProfileColumns() {
    const query = byId("column-search").value.trim().toLocaleLowerCase();
    const role = byId("semantic-role-filter").value;
    return profileColumns.filter((column) => {
      const matchesName = String(column.column_name).toLocaleLowerCase().includes(query);
      const normalizedRole = isResolvedSemantic(column.semantic_role) ? String(column.semantic_role) : "__unknown__";
      return matchesName && (!role || normalizedRole === role);
    });
  }

  function formatProfileTimestamp(value) {
    if (!value) return "Profiling time unavailable";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return "Profiling time unavailable";
    return `Profiled on ${parsed.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}`;
  }

  function renderProfilingSummary() {
    const manifest = profilingManifest || {};
    byId("profiling-dataset-name").textContent = manifest.dataset_name || (currentDataset && currentDataset.name) || "Dataset";
    byId("profiling-dataset-name").title = byId("profiling-dataset-name").textContent;
    byId("profiling-timestamp").querySelector("span").textContent = formatProfileTimestamp(manifest.profiled_at);
    const status = manifest.profile_status || "unavailable";
    byId("profiling-summary").innerHTML = `
      <article class="overview-kpi"><span class="kpi-icon purple" aria-hidden="true">${overviewIcon("table")}</span><div><span>Total Columns</span><strong>${number(manifest.total_columns).toLocaleString()}</strong><small>Analyzed in dataset</small></div></article>
      <article class="overview-kpi"><span class="kpi-icon purple" aria-hidden="true">${overviewIcon("database")}</span><div><span>Semantic Roles Detected</span><strong>${number(manifest.semantic_role_count).toLocaleString()}</strong><small>Resolved roles only</small></div></article>
      <article class="overview-kpi"><span class="kpi-icon orange" aria-hidden="true">${overviewIcon("percent")}</span><div><span>Columns Requiring Cleaning</span><strong>${number(manifest.columns_requiring_cleaning).toLocaleString()}</strong><small>Need attention</small></div></article>
      <article class="overview-kpi"><span class="kpi-icon ${status === "completed" ? "green" : status === "failed" ? "red" : "orange"}" aria-hidden="true">${status === "completed" ? "✓" : status === "failed" ? "×" : "!"}</span><div><span>Profile Status</span><strong class="profile-status-value">${escapeHtml(profileStatusLabel(status))}</strong><small>${status === "completed" ? "All columns analyzed" : status === "partial" ? "Some evidence is incomplete" : "Profiling evidence unavailable"}</small></div></article>`;
  }

  function renderProfilingWorkspace() {
    const columns = filteredProfileColumns();
    const pageCount = Math.max(1, Math.ceil(columns.length / PROFILE_PAGE_SIZE));
    profilePage = Math.min(Math.max(1, profilePage), pageCount);
    const start = (profilePage - 1) * PROFILE_PAGE_SIZE;
    const pageRows = columns.slice(start, start + PROFILE_PAGE_SIZE);
    if (columns.length && !columns.some((column) => number(column.column_index) === selectedProfileIndex)) {
      selectedProfileIndex = number(columns[0].column_index);
      selectedProfileColumn = columns[0].column_name;
    }
    const body = byId("semantic-manifest-body");
    body.innerHTML = pageRows.map((column) => {
      const selected = number(column.column_index) === selectedProfileIndex;
      const action = column.primary_action ? actionLabel(column.primary_action.action_id) : "No action required";
      return `<tr class="${selected ? "selected" : ""}" tabindex="0" aria-selected="${selected}" data-column-index="${number(column.column_index)}"><td>${number(column.column_index) + 1}</td><td class="manifest-column-name" title="${escapeHtml(column.column_name)}">${escapeHtml(column.column_name)}</td><td>${escapeHtml(column.physical_dtype || "Unknown")}</td><td>${escapeHtml(semanticLabel(column.semantic_role))}</td><td>${escapeHtml(semanticLabel(column.semantic_type))}</td><td class="manifest-action" title="${escapeHtml(action)}">${escapeHtml(action)}</td><td><span class="profile-status ${escapeHtml(column.status || "unavailable")}">${escapeHtml(profileStatusLabel(column.status || "unavailable"))}</span></td></tr>`;
    }).join("");
    byId("semantic-manifest-empty").hidden = columns.length !== 0;
    body.closest("table").hidden = columns.length === 0;
    const end = Math.min(start + pageRows.length, columns.length);
    byId("column-count").textContent = columns.length ? `Showing ${start + 1}–${end} of ${columns.length.toLocaleString()} columns` : "No matching columns";
    byId("manifest-page-prev").disabled = profilePage <= 1;
    byId("manifest-page-next").disabled = profilePage >= pageCount;
    const firstPage = Math.max(1, Math.min(profilePage - 2, pageCount - 4));
    const lastPage = Math.min(pageCount, firstPage + 4);
    byId("manifest-page-buttons").innerHTML = Array.from({ length: lastPage - firstPage + 1 }, (_, offset) => firstPage + offset).map((page) => `<button type="button" class="${page === profilePage ? "active" : ""}" data-page="${page}" aria-current="${page === profilePage ? "page" : "false"}">${page}</button>`).join("");
    renderSelectedColumn();
  }

  function selectProfileColumn(columnIndex) {
    const column = profileColumns.find((item) => number(item.column_index) === number(columnIndex));
    if (!column) return;
    selectedProfileIndex = number(column.column_index);
    selectedProfileColumn = column.column_name;
    renderProfilingWorkspace();
  }

  function renderSelectedColumn() {
    const profile = profileColumns.find((item) => number(item.column_index) === selectedProfileIndex);
    if (!profile) {
      byId("column-profile").innerHTML = `<p class="empty-inline">No column profile is available for the current filter.</p>`;
      return;
    }
    const total = number(profile.non_null_count) + number(profile.missing_count);
    const nonNullPercentage = total ? (number(profile.non_null_count) / total) * 100 : 0;
    const action = profile.primary_action;
    const actionText = action ? actionLabel(action.action_id) : "No action required";
    const explanation = profile.decision_explanation || "Explanation unavailable";
    const examples = Array.isArray(profile.observed_examples) ? profile.observed_examples : [];
    const privacyMessage = profile.privacy_restricted && !examples.length
      ? "Observed examples are withheld by the privacy policy."
      : "No safe observed examples are available.";
    const actionHistory = Array.isArray(profile.actions) && profile.actions.length > 1
      ? `<details class="action-history"><summary>View all ${profile.actions.length} decisions</summary>${profile.actions.map((item) => `<div><span>${escapeHtml(actionLabel(item.action_id))}</span><span class="profile-status ${escapeHtml(item.status)}">${escapeHtml(profileStatusLabel(item.status))}</span>${item.rationale ? `<p>${escapeHtml(item.rationale)}</p>` : ""}</div>`).join("")}</details>` : "";
    byId("column-profile").innerHTML = `
      <div class="column-title-row"><div><p class="eyebrow">Selected column</p><h2 title="${escapeHtml(profile.column_name)}">${escapeHtml(profile.column_name)}</h2></div><span class="type-tag">${escapeHtml(profile.physical_dtype || "Unknown")}</span></div>
      <div class="profile-definition"><h3>Meaning</h3><p>${escapeHtml(profile.meaning || "No semantic description was available for this column.")}</p></div>
      <div class="profile-stats"><div class="profile-stat"><span>Physical type</span><strong>${escapeHtml(profile.physical_dtype || "Unknown")}</strong></div><div class="profile-stat"><span>Semantic role</span><strong>${escapeHtml(semanticLabel(profile.semantic_role))}</strong></div><div class="profile-stat"><span>Semantic type</span><strong>${escapeHtml(semanticLabel(profile.semantic_type))}</strong></div><div class="profile-stat"><span>Non-null values</span><strong>${number(profile.non_null_count).toLocaleString()} (${nonNullPercentage.toFixed(1)}%)</strong></div><div class="profile-stat"><span>Missing values</span><strong>${number(profile.missing_count).toLocaleString()} (${number(profile.missing_percentage).toFixed(1)}%)</strong></div><div class="profile-stat"><span>Unique values</span><strong>${number(profile.unique_count).toLocaleString()}</strong></div><div class="profile-stat assigned-action"><span>Assigned action</span><strong>${escapeHtml(actionText)}</strong><span class="profile-status ${escapeHtml(profile.status || "unavailable")}">${escapeHtml(profileStatusLabel(profile.status || "unavailable"))}</span></div></div>
      ${actionHistory}
      <section class="decision-explanation"><h3>Decision Explanation</h3><p>${escapeHtml(explanation)}</p><small>Evidence source: ${escapeHtml(humanize(profile.explanation_source || "unavailable"))}</small></section>
      <section class="sample-block"><h3>Observed Examples</h3>${examples.length ? `<div class="sample-values">${examples.map((value) => `<code title="${escapeHtml(value)}">${escapeHtml(value == null ? "Missing" : value)}</code>`).join("")}</div>` : `<p class="profile-empty-copy">${escapeHtml(privacyMessage)}</p>`}</section>`;
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
    currentTaskResolution = null;
    updatePredictionSetup("initial");
    showPredictionSubView("target");
  }

  function predictionTaskLabel(resolution) {
    if (!resolution || resolution.status !== "resolved") return "Not detected";
    if (resolution.task_type === "classification") {
      if (resolution.classification_type === "binary") return "Binary Classification";
      if (resolution.classification_type === "multiclass") return "Multiclass Classification";
      return "Classification";
    }
    return resolution.task_type === "regression" ? "Regression" : "Unsupported";
  }

  function updatePredictionSetup(state, message = "") {
    const status = byId("prediction-status-label");
    const help = byId("prediction-status-help");
    const task = byId("prediction-task-type");
    const taskHelp = byId("prediction-task-help");
    const run = byId("btn-run-prediction");
    const label = byId("prediction-run-label");
    const selected = Boolean(byId("prediction-target").value);
    const states = {
      initial: ["Not Started", "Choose a target to begin"],
      resolving: ["Not Started", "Detecting the target task"],
      ready: ["Ready", "Target is valid for prediction"],
      running: ["Running", "Model training and evaluation in progress"],
      completed: ["Completed", "Model trained and evaluated"],
      failed: ["Failed", message || "Prediction could not be completed"],
      unsupported: ["Unsupported", message || "This target cannot be predicted safely"],
    };
    const content = states[state] || states.initial;
    status.textContent = content[0];
    help.textContent = content[1];
    task.textContent = state === "resolving" ? "Detecting…" : predictionTaskLabel(currentTaskResolution);
    taskHelp.textContent = currentTaskResolution && currentTaskResolution.explanation
      ? currentTaskResolution.explanation
      : (selected ? "Task detection uses the prepared target" : "Select a target to detect its task");
    run.disabled = !predictionEligible || !selected || !currentTaskResolution || currentTaskResolution.status !== "resolved" || state === "running" || state === "resolving";
    label.textContent = state === "completed" ? "Run Again" : "Run Prediction";
  }

  async function resolveSelectedPredictionTarget(target) {
    const token = ++predictionResolutionToken;
    updatePredictionSetup("resolving");
    try {
      if (!globalThis.LYNXApiClient || typeof globalThis.LYNXApiClient.resolvePredictionTarget !== "function") {
        const integrationError = new Error("Prediction target validation is temporarily unavailable. Refresh the page and try again.");
        integrationError.code = "prediction_client_incompatible";
        throw integrationError;
      }
      const resolution = await globalThis.LYNXApiClient.resolvePredictionTarget(currentRunId, target);
      if (token !== predictionResolutionToken || byId("prediction-target").value !== target) return;
      currentTaskResolution = resolution;
      if (resolution.status === "resolved") {
        byId("prediction-eligibility").textContent = "Target selected. Ready to run prediction.";
        updatePredictionSetup("ready");
      } else {
        const explanation = resolution.explanation || "This target does not support a reliable prediction task.";
        byId("prediction-eligibility").textContent = explanation;
        updatePredictionSetup("unsupported", explanation);
      }
    } catch (error) {
      if (token !== predictionResolutionToken) return;
      currentTaskResolution = null;
      const failure = predictionTargetResolutionFailure(error);
      byId("prediction-eligibility").textContent = failure.message;
      updatePredictionSetup(failure.state, failure.message);
    }
  }

  function predictionTargetResolutionFailure(error) {
    const code = error && typeof error.code === "string" ? error.code : "prediction_client_error";
    const unsupportedCodes = new Set([
      "invalid_target", "target_not_found", "task_type_unresolved",
      "target_unresolved", "prediction_not_eligible",
    ]);
    if (unsupportedCodes.has(code)) {
      return {
        state: "unsupported",
        message: error && error.message ? error.message : "This target cannot be used for prediction.",
      };
    }
    if (code === "network_error" || code === "network_unavailable") {
      return {
        state: "failed",
        message: error.message || "The LYNX service could not be reached. Check your connection and try again.",
      };
    }
    return {
      state: "failed",
      message: code === "prediction_client_incompatible"
        ? "Prediction target validation is temporarily unavailable. Refresh the page and try again."
        : "Target validation could not be completed. Please try again.",
    };
  }

  function setupPredictionBoundary() {
    setupPredictionController();

    const select = byId("prediction-target");
    const runBtn = byId("btn-run-prediction");

    select.addEventListener("change", () => {
      const selected = select.value;
      predictionResolutionToken += 1;
      if (predictionController) predictionController.cancel();
      currentPredictionTarget = selected;
      currentPredictionId = null;
      currentPredictionResult = null;
      currentTaskResolution = null;
      byId("prediction-last-run").hidden = true;
      showPredictionSubView("target");
      if (!selected) {
        byId("prediction-eligibility").textContent = predictionEligible
          ? "Select a target column to predict."
          : LYNXPipeline1Controller.eligibilityMessage(pipeline1Result).message;
        updatePredictionSetup("initial");
      } else if (!predictionEligible) {
        byId("prediction-eligibility").textContent = LYNXPipeline1Controller.eligibilityMessage(pipeline1Result).message;
        updatePredictionSetup("unsupported", byId("prediction-eligibility").textContent);
      } else {
        resolveSelectedPredictionTarget(selected);
      }
    });

    runBtn.addEventListener("click", (event) => event.preventDefault());
    runBtn.addEventListener("click", beginPrediction);

    const retryBtn = byId("btn-prediction-retry");
    if (retryBtn) {
      retryBtn.addEventListener("click", () => {
        beginPrediction();
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
    if (!target || !predictionEligible || !currentRunId || !predictionController || !currentTaskResolution || currentTaskResolution.status !== "resolved") return;
    currentPredictionTarget = target;
    byId("prediction-waiting-target-name").textContent = target;
    byId("prediction-target").disabled = true;
    byId("btn-run-prediction").disabled = true;
    updatePredictionSetup("running");
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
        updatePredictionSetup("running");
      },
      onResult(result, initialRows) {
        stopPredictionElapsed();
        currentPredictionResult = result;
        const targetSelect = byId("prediction-target");
        if (targetSelect) targetSelect.disabled = false;
        renderPredictionResults(result, initialRows);
        updatePredictionSetup("completed");
        showPredictionSubView("results");
      },
      onFailure(error) {
        stopPredictionElapsed();
        const targetSelect = byId("prediction-target");
        if (targetSelect) targetSelect.disabled = false;
        const message = error && error.message ? error.message : "Prediction could not be completed.";
        byId("prediction-error-text").textContent = message;
        updatePredictionSetup("failed", message);
        showPredictionSubView("error");
      },
    });
  }

  function showPredictionSubView(view) {
    const emptySection = byId("prediction-empty-section");
    const waitingSection = byId("prediction-waiting-section");
    const resultsSection = byId("prediction-results-section");
    const errorSection = byId("prediction-error-section");

    if (emptySection) emptySection.hidden = view !== "target";
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
    byId("pred-summary-task").textContent = predictionTaskLabel({
      status: "resolved", task_type: result.task_type,
      classification_type: result.classification_type,
    });
    if (result.completed_at) {
      const date = new Date(result.completed_at);
      if (!Number.isNaN(date.getTime())) {
        byId("prediction-last-run").hidden = false;
        byId("prediction-last-run").querySelector("span").textContent = `Last predicted ${date.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}`;
      }
    }

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
        ["Accuracy", formatPercent(primary.accuracy), "▥"],
        ["F1 Score", formatPercent(primary.f1_macro), "▤"],
        ["ROC AUC", formatPercent(primary.roc_auc), "⌁"],
        ["Balanced Accuracy", formatPercent(primary.balanced_accuracy), "⚖"],
      ].map(([label, val, icon]) => `
        <article class="metric-card"><span class="metric-icon" aria-hidden="true">${icon}</span><div>
          <span class="metric-card-label">${escapeHtml(label)}</span>
          <strong class="metric-card-value">${escapeHtml(val)}</strong>
          ${label === "F1 Score" ? "<small>Macro average</small>" : ""}
        </div></article>
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
        byId("prediction-viz-title").textContent = "Confusion Matrix";
        byId("prediction-viz-subtitle").textContent = "Rows: Actual, Columns: Predicted";
        vizSection.hidden = false;
        vizContainer.innerHTML = renderConfusionMatrix(confusion, result.class_labels);
      } else {
        vizSection.hidden = true;
      }
    } else {
      const metrics = result.metrics || {};
      metricsContainer.innerHTML = [
        ["R²", formatDecimal(metrics.r2, 4), "R²"],
        ["RMSE", formatAdaptiveNumber(metrics.rmse), "⌁"],
        ["MAE", formatAdaptiveNumber(metrics.mae), "±"],
        ["NRMSE", formatAdaptiveNumber(metrics.nrmse), "%"],
      ].map(([label, val, icon]) => `
        <article class="metric-card"><span class="metric-icon" aria-hidden="true">${icon}</span><div>
          <span class="metric-card-label">${escapeHtml(label)}</span>
          <strong class="metric-card-value">${escapeHtml(val)}</strong>
        </div></article>
      `).join("");
      secondaryWrapper.innerHTML = "";

      const vizSection = byId("prediction-viz-section");
      const vizContainer = byId("prediction-viz-container");
      const rows = initialRows && Array.isArray(initialRows.rows) ? initialRows.rows : [];
      if (rows.length) {
        byId("prediction-viz-title").textContent = "Actual vs. Predicted";
        byId("prediction-viz-subtitle").textContent = "Held-out values on the original target scale";
        vizSection.hidden = false;
        vizContainer.innerHTML = renderRegressionScatterPlot(rows);
      } else {
        vizSection.hidden = true;
      }
    }

    renderFeatureImportance(result.feature_importance);
    renderPredictionTable(isClassification, initialRows);
  }

  function renderFeatureImportance(importance) {
    const target = byId("prediction-feature-importance");
    if (!importance || importance.available !== true || !Array.isArray(importance.features) || !importance.features.length) {
      const reason = importance && importance.reason ? importance.reason : "Reliable feature importance is unavailable for this model.";
      target.innerHTML = `<div class="prediction-unavailable"><strong>Feature importance unavailable</strong><p>${escapeHtml(reason)}</p></div>`;
      return;
    }
    const values = importance.features.filter((item) => Number.isFinite(Number(item.importance)));
    const max = Math.max(...values.map((item) => Math.max(0, Number(item.importance))), 0);
    target.innerHTML = `<div class="feature-bars">${values.map((item) => {
      const value = Number(item.importance);
      const width = max > 0 ? Math.max(2, Math.max(0, value) / max * 100) : 2;
      return `<div class="feature-bar-row"><span title="${escapeHtml(item.feature)}">${escapeHtml(item.feature)}</span><div class="feature-bar-track"><i style="width:${width.toFixed(2)}%"></i></div><strong>${escapeHtml(formatAdaptiveNumber(value))}</strong></div>`;
    }).join("")}</div><p class="feature-method"><strong>Method:</strong> ${escapeHtml(importance.interpretation || humanize(importance.method))} Based on ${number(importance.sample_rows).toLocaleString()} held-out rows and ${number(importance.repeats)} permutations per feature.</p>`;
  }

  function renderConfusionMatrix(matrix, labels) {
    const numRows = matrix.length;
    const numCols = matrix[0].length;
    let html = `<div class="confusion-matrix-wrapper"><table class="confusion-matrix-table">`;
    html += `<caption>Confusion Matrix (Rows: Actual, Columns: Predicted)</caption>`;
    html += `<thead><tr><th scope="col">Actual \\ Predicted</th>`;
    const classLabels = Array.isArray(labels) && labels.length === numCols ? labels : Array.from({ length: numCols }, (_item, index) => index);
    for (let c = 0; c < numCols; c++) {
      html += `<th scope="col">${escapeHtml(classLabels[c])}</th>`;
    }
    html += `</tr></thead><tbody>`;

    for (let r = 0; r < numRows; r++) {
      html += `<tr><th scope="row">${escapeHtml(classLabels[r] != null ? classLabels[r] : r)}</th>`;
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

  function setupConversationalSurfaces() {
    insightConversation = LYNXConversationController.createConversationController({
      apiClient: LYNXApiClient,
      onChange: (state) => renderConversation("insights", state),
    });
    visualizationConversation = LYNXConversationController.createConversationController({
      apiClient: LYNXApiClient,
      onChange: (state) => renderConversation("visualization", state),
    });
    bindConversationSurface("insights", insightConversation);
    bindConversationSurface("visualization", visualizationConversation);
    byId("btn-retry-insights-landing").addEventListener("click", () => loadInsightsLanding(true));
    byId("automatic-visualization-status").addEventListener("click", (event) => {
      if (event.target.closest("[data-retry-visualizations]")) loadAutomaticVisualizations(true);
    });
  }

  function bindConversationSurface(prefix, conversation) {
    const form = byId(`${prefix}-form`);
    const input = byId(`${prefix}-input`);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const text = input.value;
      if (!text.trim()) return;
      input.value = "";
      conversation.submit(text);
      updateConversationComposer(prefix, conversation.getState());
    });
    input.addEventListener("input", () => updateConversationComposer(prefix, conversation.getState()));
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });
    const submitChoice = (event) => {
      const choice = event.target.closest("[data-conversation-choice], [data-prompt]");
      if (!choice) return;
      conversation.submit(choice.dataset.conversationChoice || choice.dataset.prompt);
    };
    byId(`${prefix}-messages`).addEventListener("click", submitChoice);
    if (prefix === "insights") byId("insights-suggested-questions").addEventListener("click", submitChoice);
  }

  function conversationStartMarkup(prefix) {
    const insight = prefix === "insights";
    if (insight) return `<div class="insights-empty-conversation"><strong>Ask LYNX about this dataset</strong><p>Choose a suggested question or enter your own factual question below.</p></div>`;
    return `<div class="custom-visualization-empty"><strong>No custom charts yet</strong><p>Describe a supported comparison, trend, relationship, or distribution.</p></div>`;
  }

  async function loadInsightsLanding(force = false) {
    if (!currentRunId || (!force && insightsLandingDataset === currentRunId)) return;
    const requestedDataset = currentRunId;
    insightsLandingDataset = requestedDataset;
    renderInsightsLanding("loading");
    try {
      const payload = await LYNXApiClient.getInsightsLanding(requestedDataset);
      if (currentRunId !== requestedDataset) return;
      insightsLanding = payload;
      renderInsightsLanding("ready");
    } catch (_error) {
      if (currentRunId === requestedDataset) {
        insightsLandingDataset = null;
        insightsLanding = null;
        renderInsightsLanding("error");
      }
    }
  }

  function renderInsightsLanding(state) {
    const summary = byId("insights-summary-copy");
    const suggestions = byId("insights-suggested-questions");
    const retry = byId("btn-retry-insights-landing");
    if (!summary || !suggestions || !retry) return;
    applySuggestionVisibility();
    retry.hidden = state !== "error";
    if (state === "loading") {
      summary.innerHTML = "<p>Preparing a grounded summary of this dataset…</p>";
      suggestions.innerHTML = '<span class="insights-loading-copy">Preparing questions supported by this dataset…</span>';
      return;
    }
    if (state === "error" || !insightsLanding) {
      summary.innerHTML = "<p>The dataset summary is temporarily unavailable. You can still ask LYNX a question below.</p>";
      suggestions.innerHTML = '<span class="insights-loading-copy">Suggested questions are temporarily unavailable.</span>';
      return;
    }
    const paragraphs = String(insightsLanding.summary_text || "")
      .split(/\n\s*\n/).map((item) => item.trim()).filter(Boolean).slice(0, 2);
    summary.innerHTML = paragraphs.length
      ? paragraphs.map((item) => `<p>${escapeHtml(item)}</p>`).join("")
      : "<p>No verified summary is available for this dataset.</p>";
    const prompts = Array.isArray(insightsLanding.suggested_questions)
      ? insightsLanding.suggested_questions.map((item) => item && item.text).filter(Boolean).slice(0, 5) : [];
    suggestions.innerHTML = prompts.length
      ? prompts.map((prompt) => `<button type="button" data-prompt="${escapeHtml(prompt)}"><svg aria-hidden="true" viewBox="0 0 24 24"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5"/></svg><span>${escapeHtml(prompt)}</span></button>`).join("")
      : '<span class="insights-loading-copy">No supported suggested questions are available for this dataset.</span>';
  }

  function renderConversation(prefix, state) {
    const feed = byId(`${prefix}-messages`);
    if (!feed) return;
    if (prefix === "visualization" && visualizationRenderer) visualizationRenderer.disposeWithin(feed);
    feed.innerHTML = state.messages.length
      ? state.messages.map((message) => conversationMessageMarkup(message, prefix)).join("")
      : conversationStartMarkup(prefix);
    if (prefix === "visualization" && visualizationRenderer) {
      const visuals = state.messages
        .filter((message) => message.role === "assistant" && message.visualization)
        .flatMap((message) => Array.isArray(message.visualization.visualizations) ? message.visualization.visualizations : []);
      visualizationRenderer.renderAll(feed, visuals);
    }
    const status = byId(`${prefix}-status`);
    status.classList.toggle("is-error", Boolean(state.error));
    status.textContent = state.loading
      ? (prefix === "visualization" ? "Generating a verified visualization…" : "Computing insight…")
      : state.error ? safeConversationError(state.error) : "";
    updateConversationComposer(prefix, state);
    if (prefix === "insights") {
      document.querySelectorAll("#insights-suggested-questions button").forEach((button) => { button.disabled = state.loading; });
    }
    if (state.messages.length) feed.scrollTop = feed.scrollHeight;
  }

  function updateConversationComposer(prefix, state) {
    const input = byId(`${prefix}-input`);
    const submit = byId(`${prefix}-form`).querySelector("button[type=submit]");
    if (!input || !submit) return;
    input.disabled = state.loading || !state.datasetId;
    submit.disabled = state.loading || !state.datasetId || !input.value.trim();
  }

  function safeConversationError(error) {
    if (["dataset_not_found", "conversation_not_found", "conversation_dataset_mismatch"].includes(error.code)) {
      return error.message;
    }
    return "Unable to process this request right now. Please try again.";
  }

  function conversationMessageMarkup(message, prefix) {
    if (prefix === "insights") return insightsMessageMarkup(message);
    return visualizationMessageMarkup(message);
  }

  function insightsMessageMarkup(message) {
    const time = formatConversationTime(message.createdAt);
    if (message.role === "user") {
      return `<article class="insights-message insights-user-message"><span class="insights-avatar user-avatar" aria-hidden="true">Y</span><div class="insights-message-body"><header><strong>You</strong>${time ? `<time>${escapeHtml(time)}</time>` : ""}</header><p>${escapeHtml(message.text)}</p></div></article>`;
    }
    const answer = message.intent === "INSIGHT" && message.insight
      ? (message.insight.answer || message.insight.message || message.text) : message.text;
    const choices = message.clarificationOptions && message.clarificationOptions.length
      ? `<div class="clarification-choices" aria-label="Clarification choices">${message.clarificationOptions.map((choice) => `<button type="button" data-conversation-choice="${escapeHtml(choice)}">${escapeHtml(choice)}</button>`).join("")}</div>` : "";
    return `<article class="insights-message insights-lynx-message assistant-message" data-intent="${escapeHtml(message.intent)}" data-status="${escapeHtml(message.status)}"><span class="insights-avatar lynx-avatar" aria-hidden="true"><svg viewBox="0 0 42 42"><path d="M8 30 15 24l6 2 6-10 7-5"/><circle cx="8" cy="30" r="2"/><circle cx="15" cy="24" r="2"/><circle cx="21" cy="26" r="2"/><circle cx="27" cy="16" r="2"/><circle cx="34" cy="11" r="2"/></svg></span><div class="insights-message-body"><header><strong>LYNX</strong>${time ? `<time>${escapeHtml(time)}</time>` : ""}</header>${answer ? `<p class="insight-answer">${escapeHtml(answer)}</p>` : ""}${choices}</div></article>`;
  }

  function formatConversationTime(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(date);
  }

  function visualizationMessageMarkup(message) {
    const time = formatConversationTime(message.createdAt);
    if (message.role === "user") {
      return `<article class="custom-viz-request-row"><span class="insights-avatar user-avatar" aria-hidden="true">Y</span><div><header><strong>You</strong>${time ? `<time>${escapeHtml(time)}</time>` : ""}</header><p>${escapeHtml(message.text)}</p></div></article>`;
    }
    const visuals = message.intent === "VISUALIZATION" && message.visualization && Array.isArray(message.visualization.visualizations)
      ? message.visualization.visualizations : [];
    if (visuals.length) {
      return `<div class="custom-visualization-response" data-status="${escapeHtml(message.status)}">${visuals.map((item, index) => visualizationCardMarkup(item, false, index, true)).join("")}</div>`;
    }
    const choices = message.clarificationOptions && message.clarificationOptions.length
      ? `<div class="clarification-choices" aria-label="Clarification choices">${message.clarificationOptions.map((choice) => `<button type="button" data-conversation-choice="${escapeHtml(choice)}">${escapeHtml(choice)}</button>`).join("")}</div>` : "";
    return `<article class="custom-viz-notice assistant-message" data-status="${escapeHtml(message.status)}"><span class="insights-avatar lynx-avatar" aria-hidden="true"><svg viewBox="0 0 42 42"><path d="M8 30 15 24l6 2 6-10 7-5"/><circle cx="8" cy="30" r="2"/><circle cx="15" cy="24" r="2"/><circle cx="21" cy="26" r="2"/><circle cx="27" cy="16" r="2"/><circle cx="34" cy="11" r="2"/></svg></span><div><strong>LYNX</strong><p>${escapeHtml(message.text || "This visualization request could not be completed.")}</p>${choices}</div></article>`;
  }

  function visualizationCardMarkup(item, ranked = true, index = 0, requested = false) {
    const explanation = item.explanation || {};
    const xai = item.xai || explanation;
    const title = item.title || (item.chart_spec && item.chart_spec.title) || (item.y_column
      ? `${humanize(item.y_column)} by ${humanize(item.x_column)}` : humanize(item.x_column));
    const source = item.source === "USER_REQUESTED" || requested ? "User-requested visual" : "Recommended visual";
    const limitations = Array.isArray(xai.limitations) ? xai.limitations.filter(Boolean).slice(0, 2) : [];
    const whatItShows = xai.what_it_shows || explanation.what_it_shows || "The chart displays the verified analytical result for this view.";
    const whyThisChart = xai.why_this_chart || explanation.why_this_chart || "This chart type matches the grounded analytical structure of the result.";
    const alternating = index % 2 ? " explanation-first" : " chart-first";
    return `<article class="visualization-story${alternating}" data-visualization-source="${escapeHtml(source)}"><section class="visualization-chart-pane"><div class="chart-card-heading"><div>${ranked && item.rank ? `<span class="chart-rank">${escapeHtml(item.rank)}</span>` : ""}<h3>${escapeHtml(title)}</h3></div><span class="type-tag">${escapeHtml(item.presentation_variant || item.chart_type)}</span></div><div class="echarts-host" data-echarts-host role="img" aria-label="${escapeHtml(title)}"></div></section><aside class="visualization-explanation"><span class="visual-source-label">${escapeHtml(source)}</span><h3>Explanation</h3><p>${escapeHtml(whatItShows)}</p><p>${escapeHtml(whyThisChart)}</p>${limitations.length ? `<p class="visualization-limit"><strong>Limitation:</strong> ${escapeHtml(limitations.join(" "))}</p>` : ""}</aside></article>`;
  }

  function renderVisualizationKpis(kpis) {
    const container = byId("visualization-kpis");
    if (!container) return;
    const items = Array.isArray(kpis) ? kpis.slice(0, 4) : [];
    container.innerHTML = items.map((item) => `<article class="visualization-kpi"><span class="visualization-kpi-icon" aria-hidden="true">${visualizationKpiIcon(item.expression && item.expression.kind)}</span><div><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(LYNXChartFormatter.formatValue(item.value, item.format_hint, { compact: true }))}</strong></div></article>`).join("");
    container.hidden = !items.length;
  }

  function visualizationKpiIcon(kind) {
    if (kind === "COUNT_ROWS" || kind === "COUNT_DISTINCT") return '<svg viewBox="0 0 24 24"><rect x="4" y="5" width="16" height="14" rx="2"/><path d="M8 9h8M8 13h8M8 17h5"/></svg>';
    if (kind === "RANK_COUNT") return '<svg viewBox="0 0 24 24"><path d="M5 19V12h3v7M11 19V8h3v11M17 19V4h3v15"/></svg>';
    return '<svg viewBox="0 0 24 24"><path d="M4 18 9 12l4 3 7-9"/><path d="M16 6h4v4"/></svg>';
  }

  function clearAutomaticVisualizations() {
    const grid = byId("visualization-grid");
    const status = byId("automatic-visualization-status");
    if (grid) {
      if (visualizationRenderer) visualizationRenderer.disposeWithin(grid);
      grid.innerHTML = "";
    }
    renderVisualizationKpis([]);
    if (status) status.textContent = "";
  }

  async function loadAutomaticVisualizations(force = false) {
    if (!currentRunId || (!force && automaticVisualizationDataset === currentRunId)) return;
    const requestedDataset = currentRunId;
    automaticVisualizationDataset = requestedDataset;
    const status = byId("automatic-visualization-status");
    const grid = byId("visualization-grid");
    status.textContent = "Loading recommended visualizations…";
    if (visualizationRenderer) visualizationRenderer.disposeWithin(grid);
    grid.innerHTML = "";
    try {
      const payload = await LYNXApiClient.getAutomaticVisualizations(requestedDataset, { maxVisualizations: 4 });
      if (currentRunId !== requestedDataset) return;
      const visualizations = Array.isArray(payload.visualizations) ? payload.visualizations : [];
      status.textContent = visualizations.length ? "" : (payload.message || "No meaningful visualization is supported by this dataset.");
      renderVisualizationKpis(payload.kpis);
      grid.innerHTML = visualizations.map((item, index) => visualizationCardMarkup(item, true, index, false)).join("");
      if (visualizationRenderer) visualizationRenderer.renderAll(grid, visualizations);
    } catch (_error) {
      if (currentRunId !== requestedDataset) return;
      automaticVisualizationDataset = null;
      status.innerHTML = `Recommended visualizations could not be loaded. <button class="text-button" type="button" data-retry-visualizations>Try again</button>`;
    }
  }

  function setupExportWorkspace() {
    const selectAll = byId("export-select-all");
    const sectionInputs = () => [...document.querySelectorAll("[data-export-section]")];
    selectAll.addEventListener("change", () => {
      sectionInputs().forEach((input) => { if (!input.disabled) input.checked = selectAll.checked; });
      renderExportPreview();
    });
    sectionInputs().forEach((input) => input.addEventListener("change", renderExportPreview));
    byId("btn-export-dataset").addEventListener("click", exportCleanedDataset);
    byId("btn-generate-report").addEventListener("click", exportAnalysisReport);
    renderExportPreview();
  }

  function exportContext() {
    const insightState = insightConversation ? insightConversation.getState() : {};
    const visualizationState = visualizationConversation ? visualizationConversation.getState() : {};
    return {
      prediction_id: currentPredictionId || undefined,
      insights_conversation_id: insightState.conversationId || undefined,
      visualization_conversation_id: visualizationState.conversationId || undefined,
    };
  }

  async function loadExportAvailability() {
    if (!currentRunId) return;
    const status = byId("report-export-status");
    status.textContent = "Checking available analysis sections…";
    try {
      const payload = await LYNXApiClient.getExportAvailability(currentRunId, exportContext());
      if (currentState !== STATES.EXPORT) return;
      const firstLoad = !exportAvailability;
      exportAvailability = payload;
      if (firstLoad) {
        const available = new Set(payload.sections.filter((item) => item.status === "available").map((item) => item.key));
        document.querySelectorAll("[data-export-section]").forEach((input) => { input.checked = available.has(input.dataset.exportSection); });
      }
      status.textContent = "";
      renderExportPreview();
    } catch (error) {
      exportAvailability = null;
      status.textContent = error && error.message ? error.message : "Export availability could not be loaded.";
      renderExportPreview();
    }
  }

  const exportDescriptions = {
    overview: "Dataset summary, quality, health, structural metrics",
    profiling: "Semantic manifest, column details, cleaning decisions",
    prediction: "Target, model performance, evaluation figures, feature importance",
    insights: "Summary and completed question-answer interactions",
    visualization: "Auto-generated and requested charts with explanations",
  };

  function renderExportPreview() {
    const statuses = new Map((exportAvailability && exportAvailability.sections || []).map((item) => [item.key, item.status]));
    const inputs = [...document.querySelectorAll("[data-export-section]")];
    inputs.forEach((input) => {
      const known = statuses.has(input.dataset.exportSection);
      const unavailable = known && statuses.get(input.dataset.exportSection) !== "available";
      input.disabled = unavailable;
      if (unavailable) input.checked = false;
    });
    const selected = inputs.filter((input) => input.checked && !input.disabled);
    const selectable = inputs.filter((input) => !input.disabled);
    const all = byId("export-select-all");
    all.checked = selectable.length > 0 && selected.length === selectable.length;
    all.indeterminate = selected.length > 0 && selected.length < selectable.length;
    byId("export-selection-count").textContent = `${selected.length} of ${selectable.length} available sections selected`;
    byId("btn-generate-report").disabled = !exportAvailability || !selected.length;
    const list = byId("export-preview-list");
    list.innerHTML = inputs.map((input) => {
      const key = input.dataset.exportSection;
      const available = !input.disabled;
      const included = available && input.checked;
      const state = !available ? "Unavailable" : included ? "Included" : "Excluded";
      return `<div class="export-preview-row"><span class="export-preview-symbol" aria-hidden="true">${key === "visualization" ? "▥" : key === "prediction" ? "⌁" : key === "insights" ? "◉" : key === "profiling" ? "▤" : "▥"}</span><strong>${escapeHtml(humanize(key))}</strong><span>${escapeHtml(exportDescriptions[key])}</span><em class="export-preview-status ${state.toLowerCase()}">${state}</em></div>`;
    }).join("");
  }

  function saveDownload(download) {
    const url = URL.createObjectURL(download.blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = download.filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  async function exportCleanedDataset() {
    const button = byId("btn-export-dataset");
    const status = byId("dataset-export-status");
    const format = document.querySelector('input[name="dataset-export-format"]:checked').value;
    button.disabled = true; button.setAttribute("aria-busy", "true"); status.textContent = "Preparing download…";
    try {
      saveDownload(await LYNXApiClient.downloadCleanedDataset(currentRunId, format));
      status.textContent = "Dataset download started.";
    } catch (error) {
      status.textContent = error && error.message ? error.message : "Dataset export failed.";
    } finally {
      button.disabled = false; button.removeAttribute("aria-busy");
    }
  }

  async function exportAnalysisReport() {
    const button = byId("btn-generate-report");
    const status = byId("report-export-status");
    const sections = [...document.querySelectorAll("[data-export-section]:checked:not(:disabled)")].map((input) => input.dataset.exportSection);
    const format = document.querySelector('input[name="report-export-format"]:checked').value;
    if (!sections.length) return;
    button.disabled = true; button.setAttribute("aria-busy", "true"); status.textContent = "Generating report…";
    try {
      saveDownload(await LYNXApiClient.generateAnalysisReport(currentRunId, { format, sections, ...exportContext() }));
      status.textContent = "Report download started.";
    } catch (error) {
      status.textContent = error && error.message ? error.message : "Report generation failed.";
    } finally {
      button.removeAttribute("aria-busy"); renderExportPreview();
    }
  }

  window.LYNX = {
    STATES,
    showState,
    showCompletedAnalysis,
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
    getInsightConversation: () => insightConversation,
    getVisualizationConversation: () => visualizationConversation,
  };
})();
