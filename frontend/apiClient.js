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

  return {
    LynxApiError,
    requestJson,
    createRun,
    getRunStatus,
    getPipeline1Result,
    createPrediction,
    getPredictionStatus,
    getPredictionResult,
    getPredictionRows,
  };
});
