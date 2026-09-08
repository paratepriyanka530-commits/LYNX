(function (root, factory) {
  const adapter = factory();
  if (typeof module === "object" && module.exports) module.exports = adapter;
  root.LYNXReportAdapter = adapter;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  // Product placeholder: keep configurable until the cardinality ceiling is finalized.
  const DEFAULT_CARDINALITY_CEILING = 100;
  const RISK_RANK = { high: 4, medium: 3, low: 2, context_dependent: 1 };
  const STATUS_COLORS = {
    issue: "#D94B55",
    candidate: "#E9A23B",
    observation: "#547DC3"
  };

  function humanize(value) {
    if (!value) return "";
    const text = String(value).replace(/_/g, " ").trim();
    return text ? text.charAt(0).toUpperCase() + text.slice(1) : "";
  }

  function formatBytes(value) {
    const bytes = Number(value);
    if (!Number.isFinite(bytes) || bytes < 0) return null;
    if (bytes < 1024) return `${bytes} B`;
    const units = ["KB", "MB", "GB", "TB"];
    let amount = bytes / 1024;
    let index = 0;
    while (amount >= 1024 && index < units.length - 1) {
      amount /= 1024;
      index += 1;
    }
    return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${units[index]}`;
  }

  function formatExample(value) {
    if (typeof value === "string") return value;
    try {
      return JSON.stringify(value);
    } catch (_error) {
      return String(value);
    }
  }

  function issueDescription(issue) {
    const definition = issue && issue.evidence && issue.evidence.definition;
    if (typeof definition === "string" && definition.trim()) return definition.trim();
    const examples = Array.isArray(issue && issue.examples) ? issue.examples.slice(0, 3) : [];
    if (!examples.length) return null;
    return `Examples: ${examples.map(formatExample).join(", ")}`;
  }

  function mapIssue(issue) {
    return {
      id: issue.issue_id,
      name: humanize(issue.issue_type),
      riskHint: issue.risk_hint,
      percentage: Number(issue.affected_percentage) || 0,
      affectedCount: Number(issue.affected_count) || 0,
      description: issueDescription(issue)
    };
  }

  function statusBreakdown(report) {
    const counts = (report.quality_summary && report.quality_summary.issues_by_status) || {};
    const keys = ["issue", "candidate", "observation"];
    const total = keys.reduce((sum, key) => sum + (Number(counts[key]) || 0), 0);
    return keys.map((key) => {
      const count = Number(counts[key]) || 0;
      return {
        key,
        label: humanize(key),
        count,
        percentage: total ? (count / total) * 100 : 0,
        color: STATUS_COLORS[key]
      };
    });
  }

  function boundedCardinality(report, ceiling) {
    const semantics = new Map(
      (((report.semantic_profile || {}).columns) || []).map((column) => [column.column_name, column])
    );
    const profiles = (((report.before_profile || {}).column_profiles) || []);
    const candidates = profiles.filter((profile) => {
      const semantic = semantics.get(profile.column_name) || {};
      const uniqueCount = Number(profile.uniqueness && profile.uniqueness.unique_count);
      return semantic.semantic_type === "categorical"
        && Number.isFinite(uniqueCount)
        && uniqueCount >= 2
        && uniqueCount <= ceiling;
    });
    candidates.sort((left, right) => {
      const difference = right.uniqueness.unique_count - left.uniqueness.unique_count;
      return difference || String(left.column_name).localeCompare(String(right.column_name));
    });
    if (!candidates.length) return null;
    return {
      label: candidates[0].column_name,
      value: `${candidates[0].uniqueness.unique_count} unique values`
    };
  }

  function basename(path) {
    return String(path || "").split(/[\\/]/).pop() || null;
  }

  function resolveArtifactPath(embeddedPath, reportLocation) {
    const filename = basename(embeddedPath);
    if (!filename || !reportLocation) return null;
    try {
      return new URL(filename, reportLocation).toString();
    } catch (_error) {
      const normalized = String(reportLocation).replace(/\\/g, "/");
      const slash = normalized.lastIndexOf("/");
      return slash >= 0 ? `${normalized.slice(0, slash + 1)}${filename}` : null;
    }
  }

  function adaptCleaningReport(report, options) {
    if (!report || typeof report !== "object" || Array.isArray(report)) {
      throw new TypeError("A Pipeline 1 cleaning report object is required.");
    }
    if (!report.original_dataset || !report.quality_summary || !report.semantic_profile) {
      throw new TypeError("The JSON does not match the Pipeline 1 cleaning report contract.");
    }

    const settings = options || {};
    const ceiling = Number.isFinite(settings.cardinalityCeiling)
      ? settings.cardinalityCeiling
      : DEFAULT_CARDINALITY_CEILING;
    const context = report.semantic_profile.dataset_context || {};
    const rowEntity = context.row_entity && context.row_entity !== "unknown"
      ? String(context.row_entity).trim()
      : null;
    const summary = context.dataset_description
      ? String(context.dataset_description).trim()
      : (rowEntity ? `Each row represents a ${rowEntity}.` : null);
    const issues = Array.isArray(report.issues) ? report.issues.map(mapIssue) : [];
    const sortedIssues = [...issues].sort((left, right) => {
      const rankDifference = (RISK_RANK[right.riskHint] || 0) - (RISK_RANK[left.riskHint] || 0);
      return rankDifference || right.percentage - left.percentage;
    });
    const issueCount = Number(report.quality_summary.total_issues_detected) || 0;
    const after = (report.validation && report.validation.after_summary) || {};
    const beforeShape = (report.before_profile && report.before_profile.shape) || {};
    const fileSize = formatBytes(report.original_dataset.file_size_bytes);
    const bounded = boundedCardinality(report, ceiling);
    const actions = (Array.isArray(report.decisions) ? report.decisions : [])
      .filter((decision) => decision.decision_type === "action")
      .map((decision) => ({
        id: decision.decision_id,
        title: humanize(decision.proposed_action),
        explanation: decision.rationale,
        confidence: Math.round((Number(decision.confidence) || 0) * 100),
        riskLevel: decision.risk_level,
        requiresReview: Boolean(decision.requires_review)
      }));
    const artifacts = report.output_artifacts || {};

    return {
      id: report.run_id,
      name: report.original_dataset.file_name || "Untitled dataset",
      domain: context.probable_domain || null,
      status: report.status,
      statusText: report.status === "success" ? "Analysis complete" : "Analysis failed",
      analysisSummary: `${issueCount} issues detected — ${report.status === "success" ? "analysis complete" : "analysis failed"}`,
      metrics: {
        rows: Number.isFinite(Number(after.row_count)) ? Number(after.row_count) : Number(beforeShape.rows) || 0,
        columns: Number.isFinite(Number(after.column_count)) ? Number(after.column_count) : Number(beforeShape.columns) || 0,
        issuesCount: issueCount
      },
      issueStatusBreakdown: statusBreakdown(report),
      allIssues: sortedIssues,
      topIssues: sortedIssues.slice(0, 4),
      summary,
      metaChips: [
        fileSize ? { label: "File Size", value: fileSize } : null,
        bounded
      ].filter(Boolean),
      boundedCardinalityCeiling: ceiling,
      processingStatus: report.status,
      recommendedActions: actions,
      outputArtifacts: {
        cleanedDataset: resolveArtifactPath(artifacts.cleaned_dataset, settings.reportLocation),
        cleaningReport: settings.reportLocation || null
      },
      visibility: {
        domain: Boolean(context.probable_domain),
        summary: Boolean(summary),
        boundedCardinality: Boolean(bounded)
      }
    };
  }

  return {
    DEFAULT_CARDINALITY_CEILING,
    adaptCleaningReport,
    resolveArtifactPath
  };
});
