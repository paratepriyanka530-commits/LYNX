// LYNX Pipeline 1 Results Dashboard

let currentDataset = null;
let currentReport = null;

document.addEventListener("DOMContentLoaded", async () => {
  setupTabNavigation();
  setupHeaderActions();

  const datasetDropdown = document.getElementById("dataset-select-dropdown");
  if (datasetDropdown) {
    datasetDropdown.addEventListener("change", async (event) => {
      await loadReportSource(event.target.value);
    });
  }

  const customJsonBtn = document.getElementById("btn-custom-json-load");
  if (customJsonBtn) {
    customJsonBtn.addEventListener("click", () => {
      const sampleJSON = JSON.stringify(currentReport || {}, null, 2);
      openModal(
        "Load Pipeline 1 cleaning_report.json",
        `<p style="margin-bottom: 10px;">Paste a Pipeline 1 cleaning report JSON payload:</p>
         <textarea id="custom-json-textarea" style="width: 100%; height: 220px; font-family: monospace; font-size: 12px; padding: 10px; border: 1px solid #DEDCE2; border-radius: 6px; background-color: #F8F7F2; color: #160B2E;">${escapeHtml(sampleJSON)}</textarea>`,
        "Apply Report",
        () => {
          const textarea = document.getElementById("custom-json-textarea");
          if (!textarea) return;
          try {
            currentReport = JSON.parse(textarea.value);
            currentDataset = LYNXReportAdapter.adaptCleaningReport(currentReport);
            renderDataset(currentDataset);
          } catch (error) {
            alert(`Invalid Pipeline 1 report: ${error.message}`);
          }
        }
      );
    });
  }

  const initialSource = datasetDropdown && datasetDropdown.value
    ? datasetDropdown.value
    : Object.keys(LYNX_REPORT_SOURCES)[0];
  await loadReportSource(initialSource);
});

async function loadReportSource(sourceId) {
  const source = LYNX_REPORT_SOURCES[sourceId];
  if (!source) return;
  try {
    const response = await fetch(source.reportUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    currentReport = await response.json();
    const reportLocation = new URL(source.reportUrl, window.location.href).toString();
    currentDataset = LYNXReportAdapter.adaptCleaningReport(currentReport, { reportLocation });
    renderDataset(currentDataset);
  } catch (error) {
    alert(`Unable to load ${source.label}: ${error.message}`);
  }
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatPercentage(value) {
  const number = Number(value) || 0;
  return `${number.toFixed(2).replace(/\.00$/, "").replace(/(\.\d)0$/, "$1")}%`;
}

function renderDataset(data) {
  if (!data) return;

  const nameEl = document.getElementById("display-dataset-name");
  const catEl = document.getElementById("display-dataset-category");
  const statusEl = document.getElementById("display-dataset-status");
  if (nameEl) nameEl.textContent = data.name;
  if (catEl) {
    catEl.textContent = data.domain || "";
    catEl.style.display = data.visibility.domain ? "" : "none";
    if (catEl.nextElementSibling) catEl.nextElementSibling.style.display = data.visibility.domain ? "" : "none";
  }
  if (statusEl) statusEl.textContent = data.statusText;

  renderMetrics(data);
  renderIssueStatusDonut(data);
  renderTopIssues(data);
  renderSummary(data);
  renderProcessingStatus(data);
  renderRecommendations(data);
}

function renderMetrics(data) {
  const metrics = data.metrics;
  const metricsRow = document.querySelector(".metrics-row");
  if (!metricsRow) return;
  const cards = [
    `<div class="metric-card">
       <span class="metric-label">Pipeline Status</span>
       <div class="metric-value-row">
         <span class="metric-value metric-value-compact">${escapeHtml(data.status)}</span>
         <span class="metric-badge ${data.status === "success" ? "badge-success" : "badge-warning"}">${escapeHtml(metrics.issuesCount)} issues detected</span>
       </div>
     </div>`,
    `<div class="metric-card"><span class="metric-label">Rows</span><div class="metric-value-row"><span class="metric-value">${metrics.rows.toLocaleString()}</span></div></div>`,
    `<div class="metric-card"><span class="metric-label">Columns</span><div class="metric-value-row"><span class="metric-value">${metrics.columns.toLocaleString()}</span></div></div>`,
    `<div class="metric-card"><span class="metric-label">Issues</span><div class="metric-value-row"><span class="metric-value">${metrics.issuesCount.toLocaleString()}</span></div></div>`
  ];
  if (data.visibility.domain) {
    cards.push(`<div class="metric-card"><span class="metric-label">Domain</span><div class="metric-value-row"><span class="metric-value metric-value-compact">${escapeHtml(data.domain)}</span></div></div>`);
  }
  metricsRow.style.gridTemplateColumns = `repeat(${cards.length}, 1fr)`;
  metricsRow.innerHTML = cards.join("");
}

function renderIssueStatusDonut(data) {
  const circumference = 251.3;
  let offset = 0;
  const donutSvg = document.getElementById("quality-donut-svg");
  if (donutSvg) {
    donutSvg.innerHTML = data.issueStatusBreakdown.map((item) => {
      const dash = (item.percentage / 100) * circumference;
      const circle = `<circle cx="50" cy="50" r="40" fill="transparent" stroke="${item.color}" stroke-width="14" stroke-dasharray="${dash} ${circumference}" stroke-dashoffset="${-offset}" />`;
      offset += dash;
      return circle;
    }).join("");
  }
  const centerVal = document.querySelector(".donut-center-value");
  const centerLabel = document.querySelector(".donut-center-subtext");
  if (centerVal) centerVal.textContent = data.metrics.issuesCount.toLocaleString();
  if (centerLabel) centerLabel.textContent = "Detected Issues";

  const legend = document.querySelector(".donut-legend-list");
  if (legend) {
    legend.innerHTML = data.issueStatusBreakdown.map((item) => `
      <div class="legend-item">
        <div class="legend-left"><span class="legend-color-dot" style="background-color: ${item.color};"></span><span class="legend-title">${escapeHtml(item.label)}</span></div>
        <div class="legend-stats"><span class="legend-count">${item.count.toLocaleString()} issues</span><span class="legend-percentage">${item.percentage.toFixed(1)}%</span></div>
      </div>`).join("");
  }
}

function issueMarkup(issue) {
  const description = issue.description
    ? `<p class="issue-desc">${escapeHtml(issue.description)}</p>`
    : "";
  return `<div class="issue-item">
    <span class="severity-pill severity-${escapeHtml(issue.riskHint)}">${escapeHtml(issue.riskHint)}</span>
    <div class="issue-details">
      <div class="issue-title-row"><span class="issue-name">${escapeHtml(issue.name)}</span><span class="issue-percentage">${formatPercentage(issue.percentage)}</span></div>
      ${description}
    </div>
  </div>`;
}

function renderTopIssues(data) {
  const topIssuesCard = document.querySelector('.bento-card[aria-labelledby="heading-top-issues"]');
  const qualityCard = document.querySelector('.bento-card[aria-labelledby="heading-quality-overview"]');
  const hasIssues = data.topIssues.length > 0;
  if (topIssuesCard) topIssuesCard.classList.toggle("card-hidden", !hasIssues);
  if (qualityCard) {
    qualityCard.classList.toggle("col-span-12", !hasIssues);
    qualityCard.classList.toggle("col-span-7", hasIssues);
  }
  const issuesContainer = document.querySelector(".issues-list");
  if (issuesContainer) issuesContainer.innerHTML = data.topIssues.map(issueMarkup).join("");
  const viewAll = document.getElementById("view-all-issues-btn");
  if (viewAll) {
    viewAll.innerHTML = `View all ${data.metrics.issuesCount.toLocaleString()} issues <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>`;
  }
}

function renderSummary(data) {
  const summaryCard = document.querySelector('.bento-card[aria-labelledby="heading-dataset-summary"]');
  if (summaryCard) summaryCard.classList.toggle("card-hidden", !data.visibility.summary);
  const summaryPara = document.querySelector(".summary-paragraph");
  if (summaryPara) summaryPara.textContent = data.summary || "";
  const chipsGrid = document.querySelector(".meta-chips-grid");
  if (chipsGrid) {
    chipsGrid.innerHTML = data.metaChips.map((chip) => `<div class="meta-chip">${escapeHtml(chip.label)}: <strong>${escapeHtml(chip.value)}</strong></div>`).join("");
  }
}

function renderProcessingStatus(data) {
  const container = document.querySelector(".status-checklist");
  if (!container) return;
  const success = data.processingStatus === "success";
  container.innerHTML = `<div class="checklist-item"><span class="check-icon-circle">${success ? "✓" : "!"}</span><span>Pipeline ${escapeHtml(data.processingStatus)}</span></div>`;
}

function renderRecommendations(data) {
  const grid = document.querySelector(".recommendations-grid");
  if (!grid) return;
  grid.innerHTML = data.recommendedActions.map((action) => {
    const reviewLabel = action.requiresReview ? "Review required" : "No review required";
    return `<div class="recommendation-item-card">
      <h3 class="rec-title">${escapeHtml(action.title)}</h3>
      <p class="rec-explanation">${escapeHtml(action.explanation)}</p>
      <div class="rec-meta-row">
        <span class="confidence-tag">${action.confidence}% Confidence</span>
        <span class="impact-tag">${escapeHtml(action.riskLevel)} risk · ${reviewLabel}</span>
      </div>
      <button class="btn-secondary-sm review-rec-btn" disabled aria-disabled="true">Coming soon</button>
    </div>`;
  }).join("");
}

function setupTabNavigation() {
  const tabs = document.querySelectorAll(".nav-tab-item");
  const panels = document.querySelectorAll(".tab-panel");
  tabs.forEach((tab) => tab.addEventListener("click", () => {
    tabs.forEach((item) => {
      item.classList.remove("active");
      item.setAttribute("aria-selected", "false");
    });
    panels.forEach((panel) => panel.classList.remove("active"));
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    const target = document.getElementById(tab.getAttribute("aria-controls"));
    if (target) target.classList.add("active");
  }));
}

function setupHeaderActions() {
  const exportBtn = document.getElementById("export-report-btn");
  if (exportBtn) {
    exportBtn.addEventListener("click", () => {
      if (!currentDataset) return;
      openModal(
        `Export Intelligence Report: ${escapeHtml(currentDataset.name)}`,
        `<p>The loaded Pipeline 1 JSON report is the authoritative report for <strong>${escapeHtml(currentDataset.name)}</strong>.</p>`,
        "Close"
      );
    });
  }

  const viewIssuesBtn = document.getElementById("view-all-issues-btn");
  if (viewIssuesBtn) {
    viewIssuesBtn.addEventListener("click", () => {
      if (!currentDataset) return;
      openModal(
        `All ${currentDataset.metrics.issuesCount} Dataset Issues (${escapeHtml(currentDataset.name)})`,
        `<div style="display: flex; flex-direction: column; gap: 10px; max-height: 300px; overflow-y: auto;">${currentDataset.allIssues.map(issueMarkup).join("")}</div>`,
        "Close"
      );
    });
  }
}

function openModal(title, bodyHTML, confirmText = "Confirm", onConfirm = null) {
  const overlay = document.getElementById("dashboard-modal");
  const titleEl = document.getElementById("modal-title-text");
  const bodyEl = document.getElementById("modal-body-text");
  const confirmBtn = document.getElementById("modal-confirm-btn");
  if (titleEl) titleEl.textContent = title;
  if (bodyEl) bodyEl.innerHTML = bodyHTML;
  if (confirmBtn) confirmBtn.textContent = confirmText;
  if (overlay) overlay.classList.add("active");
  const close = () => overlay && overlay.classList.remove("active");
  const closeX = document.getElementById("modal-close-x");
  const cancelBtn = document.getElementById("modal-cancel-btn");
  if (closeX) closeX.onclick = close;
  if (cancelBtn) cancelBtn.onclick = close;
  if (confirmBtn) confirmBtn.onclick = () => {
    if (onConfirm) onConfirm();
    close();
  };
}
