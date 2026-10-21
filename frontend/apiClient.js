/** Small same-origin client for the LYNX application API. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.LYNXApiClient = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  class LynxApiError extends Error {
    constructor(message, options = {}) {
      super(message);
      this.name = "LynxApiError";
      this.code = options.code || "request_failed";
      this.status = options.status || null;
    }
  }

  async function requestJson(path, options = {}) {
    const fetchImpl = options.fetchImpl || globalThis.fetch;
    if (typeof fetchImpl !== "function") {
      throw new LynxApiError("Network access is unavailable.", { code: "network_unavailable" });
    }

    let response;
    try {
      response = await fetchImpl(`${options.apiBase || ""}${path}`, options.request || {});
    } catch (_error) {
      throw new LynxApiError("The LYNX service could not be reached. Check your connection and try again.", {
        code: "network_error",
      });
    }

    let payload = null;
    try {
      payload = await response.json();
    } catch (_error) {
      // Never expose an HTML error page or another raw server response.
    }

    if (!response.ok) {
      const controlled = payload && payload.error;
      throw new LynxApiError(
        controlled && typeof controlled.message === "string"
          ? controlled.message
          : `The LYNX service rejected the request (HTTP ${response.status}).`,
        {
          code: controlled && controlled.code ? controlled.code : "http_error",
          status: response.status,
        }
      );
    }
    return payload;
  }

  async function requestDownload(path, options = {}) {
    const fetchImpl = options.fetchImpl || globalThis.fetch;
    if (typeof fetchImpl !== "function") {
      throw new LynxApiError("Network access is unavailable.", { code: "network_unavailable" });
    }
    let response;
    try {
      response = await fetchImpl(`${options.apiBase || ""}${path}`, options.request || {});
    } catch (_error) {
      throw new LynxApiError("The LYNX service could not be reached. Check your connection and try again.", { code: "network_error" });
    }
    if (!response.ok) {
      let payload = null;
      try { payload = await response.json(); } catch (_error) {}
      const controlled = payload && payload.error;
      throw new LynxApiError(
        controlled && controlled.message ? controlled.message : `The LYNX service rejected the request (HTTP ${response.status}).`,
        { code: controlled && controlled.code ? controlled.code : "http_error", status: response.status }
      );
    }
    const disposition = response.headers && response.headers.get ? response.headers.get("Content-Disposition") : "";
    const match = /filename="?([^";]+)"?/i.exec(disposition || "");
    return { blob: await response.blob(), filename: match ? match[1] : "lynx-export" };
  }

  function createRun(file, options = {}) {
    if (!file || typeof file !== "object") {
      return Promise.reject(new LynxApiError("Choose a valid dataset file first.", { code: "missing_file" }));
    }
    const FormDataClass = options.FormDataClass || globalThis.FormData;
    const form = new FormDataClass();
    form.append("dataset", file);
    return requestJson("/api/runs", {
      ...options,
      request: { method: "POST", body: form, signal: options.signal },
    });
  }

  function getRunStatus(runId, options = {}) {
    return requestJson(`/api/runs/${encodeURIComponent(runId)}`, {
      ...options,
      request: { method: "GET", signal: options.signal },
    });
  }

  function getPipeline1Result(runId, options = {}) {
    return requestJson(`/api/runs/${encodeURIComponent(runId)}/pipeline1`, {
      ...options,
      request: { method: "GET", signal: options.signal },
    });
  }

  function getApplicationInfo(options = {}) {
    return requestJson("/api/application-info", {
      ...options,
      request: { method: "GET", signal: options.signal },
    });
  }

  function createPrediction(runId, targetColumn, options = {}) {
    if (!runId || typeof runId !== "string") {
      return Promise.reject(new LynxApiError("Run identifier is required.", { code: "invalid_run_id" }));
    }
    if (!targetColumn || typeof targetColumn !== "string" || !targetColumn.trim()) {
      return Promise.reject(new LynxApiError("Target column must be a non-empty string.", { code: "invalid_target" }));
    }
    return requestJson(`/api/runs/${encodeURIComponent(runId)}/predictions`, {
      ...options,
      request: {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_column: targetColumn.trim() }),
        signal: options.signal,
      },
    });
  }

  function resolvePredictionTarget(runId, targetColumn, options = {}) {
    if (!runId || typeof runId !== "string" || !targetColumn || typeof targetColumn !== "string") {
      return Promise.reject(new LynxApiError("Run identifier and target column are required.", { code: "invalid_target" }));
    }
    return requestJson(`/api/runs/${encodeURIComponent(runId)}/prediction-target-resolution`, {
      ...options,
      request: {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_column: targetColumn.trim() }),
        signal: options.signal,
      },
    });
  }

  function getPredictionStatus(runId, predictionId, options = {}) {
    if (!runId || !predictionId) {
      return Promise.reject(new LynxApiError("Run ID and prediction ID are required.", { code: "missing_ids" }));
    }
    return requestJson(
      `/api/runs/${encodeURIComponent(runId)}/predictions/${encodeURIComponent(predictionId)}`,
      {
        ...options,
        request: { method: "GET", signal: options.signal },
      }
    );
  }

  function getPredictionResult(runId, predictionId, options = {}) {
    if (!runId || !predictionId) {
      return Promise.reject(new LynxApiError("Run ID and prediction ID are required.", { code: "missing_ids" }));
    }
    return requestJson(
      `/api/runs/${encodeURIComponent(runId)}/predictions/${encodeURIComponent(predictionId)}/result`,
      {
        ...options,
        request: { method: "GET", signal: options.signal },
      }
    );
  }

  function getPredictionRows(runId, predictionId, { offset = 0, limit = 50, ...options } = {}) {
    if (!runId || !predictionId) {
      return Promise.reject(new LynxApiError("Run ID and prediction ID are required.", { code: "missing_ids" }));
    }
    const params = new URLSearchParams({
      offset: String(offset),
      limit: String(limit),
    });
    return requestJson(
      `/api/runs/${encodeURIComponent(runId)}/predictions/${encodeURIComponent(predictionId)}/rows?${params.toString()}`,
      {
        ...options,
        request: { method: "GET", signal: options.signal },
      }
    );
  }

  function getDatasetPage(runId, params = {}, options = {}) {
    let run;
    try { run = validText(runId, "Run identifier is required."); }
    catch (error) { return Promise.reject(error); }
    const query = new URLSearchParams({
      version: params.version === "original" ? "original" : "cleaned",
      offset: String(Number.isInteger(params.offset) && params.offset >= 0 ? params.offset : 0),
      limit: String([10, 20, 50, 100].includes(params.limit) ? params.limit : 20),
    });
    ["search", "sort_column", "sort_direction", "filter_column", "filter_value"].forEach((key) => {
      const value = params[key];
      if (typeof value === "string" && value.trim()) query.set(key, value.trim());
    });
    return requestJson(`/api/runs/${encodeURIComponent(run)}/dataset?${query.toString()}`, {
      ...options,
      request: { method: "GET", signal: options.signal },
    });
  }

  function validText(value, message) {
    if (!value || typeof value !== "string" || !value.trim()) {
      throw new LynxApiError(message, { code: "invalid_request" });
    }
    return value.trim();
  }

  function createConversation(datasetId, message, options = {}) {
    let dataset;
    let text;
    try {
      dataset = validText(datasetId, "Dataset identifier is required.");
      text = validText(message, "Conversation message must be a non-empty string.");
    } catch (error) { return Promise.reject(error); }
    return requestJson(`/api/conversations/${encodeURIComponent(dataset)}`, {
      ...options,
      request: {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
        signal: options.signal,
      },
    });
  }

  function sendConversationMessage(datasetId, conversationId, message, options = {}) {
    let dataset;
    let conversation;
    let text;
    try {
      dataset = validText(datasetId, "Dataset identifier is required.");
      conversation = validText(conversationId, "Conversation identifier is required.");
      text = validText(message, "Conversation message must be a non-empty string.");
    } catch (error) { return Promise.reject(error); }
    return requestJson(`/api/conversations/${encodeURIComponent(dataset)}/${encodeURIComponent(conversation)}`, {
      ...options,
      request: {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
        signal: options.signal,
      },
    });
  }

  function getAutomaticVisualizations(datasetId, options = {}) {
    let dataset;
    try { dataset = validText(datasetId, "Dataset identifier is required."); }
    catch (error) { return Promise.reject(error); }
    const requestedMaximum = Number(options.maxVisualizations);
    const maximum = Number.isInteger(requestedMaximum) && requestedMaximum >= 1 && requestedMaximum <= 4
      ? `?max_visualizations=${requestedMaximum}` : "";
    return requestJson(`/api/visualizations/${encodeURIComponent(dataset)}${maximum}`, {
      ...options,
      request: { method: "GET", signal: options.signal },
    });
  }

  function getInsightsLanding(datasetId, options = {}) {
    let dataset;
    try { dataset = validText(datasetId, "Dataset identifier is required."); }
    catch (error) { return Promise.reject(error); }
    return requestJson(`/api/insights/${encodeURIComponent(dataset)}/landing`, {
      ...options,
      request: { method: "GET", signal: options.signal },
    });
  }

  function exportContext(params = {}) {
    const query = new URLSearchParams();
    ["prediction_id", "insights_conversation_id", "visualization_conversation_id"].forEach((key) => {
      if (typeof params[key] === "string" && params[key]) query.set(key, params[key]);
    });
    return query;
  }

  function getExportAvailability(runId, params = {}, options = {}) {
    let run;
    try { run = validText(runId, "Run identifier is required."); }
    catch (error) { return Promise.reject(error); }
    const query = exportContext(params);
    return requestJson(`/api/runs/${encodeURIComponent(run)}/exports/availability${query.size ? `?${query}` : ""}`, {
      ...options, request: { method: "GET", signal: options.signal },
    });
  }

  function downloadCleanedDataset(runId, format, options = {}) {
    let run;
    try { run = validText(runId, "Run identifier is required."); }
    catch (error) { return Promise.reject(error); }
    const selected = format === "csv" ? "csv" : "xlsx";
    return requestDownload(`/api/runs/${encodeURIComponent(run)}/exports/dataset?format=${selected}`, {
      ...options, request: { method: "GET", signal: options.signal },
    });
  }

  function generateAnalysisReport(runId, payload, options = {}) {
    let run;
    try { run = validText(runId, "Run identifier is required."); }
    catch (error) { return Promise.reject(error); }
    return requestDownload(`/api/runs/${encodeURIComponent(run)}/exports/report`, {
      ...options,
      request: {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload), signal: options.signal,
      },
    });
  }

  return {
    LynxApiError,
    requestJson,
    createRun,
    getRunStatus,
    getPipeline1Result,
    getApplicationInfo,
    resolvePredictionTarget,
    createPrediction,
    getPredictionStatus,
    getPredictionResult,
    getPredictionRows,
    getDatasetPage,
    createConversation,
    sendConversationMessage,
    getInsightsLanding,
    getAutomaticVisualizations,
    getExportAvailability,
    downloadCleanedDataset,
    generateAnalysisReport,
  };
});
