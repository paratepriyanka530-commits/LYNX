/** Run lifecycle controller for the real Pipeline 1 application API. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.LYNXPipeline1Controller = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const DEFAULT_POLL_INTERVAL_MS = 2000;
  const MAX_TRANSIENT_POLL_ERRORS = 3;

  function eligibilityMessage(result) {
    const targetColumns = result && result.prepared_dataset && result.prepared_dataset.target_columns;
    const eligibility = result && result.prediction_eligibility;
    if (!Array.isArray(targetColumns) || targetColumns.length === 0) {
      return { enabled: false, message: "Prediction is unavailable because no target columns were prepared." };
    }
    if (!eligibility || eligibility.eligible !== true) {
      const reasons = eligibility && Array.isArray(eligibility.reasons) ? eligibility.reasons : [];
      const joined = reasons.join(" ").toLowerCase();
      let message = "Prediction is unavailable because this prepared dataset did not pass eligibility checks.";
      if (joined.includes("privacy")) message = "Prediction is unavailable until privacy clearance passes.";
      else if (joined.includes("hash") || joined.includes("integrity")) message = "Prediction is unavailable because prepared-dataset integrity could not be verified.";
      else if (joined.includes("duplicate")) message = "Prediction is unavailable because duplicate column names remain.";
      return { enabled: false, message };
    }
    return { enabled: true, message: "This prepared dataset is eligible for the future prediction stage." };
  }

  function createPipeline1Controller(options) {
    const apiClient = options.apiClient;
    const schedule = options.schedule || ((callback, delay) => setTimeout(callback, delay));
    const cancelSchedule = options.cancelSchedule || ((id) => clearTimeout(id));
    const pollIntervalMs = options.pollIntervalMs || DEFAULT_POLL_INTERVAL_MS;
    const callbacks = {
      onRunAccepted: options.onRunAccepted || (() => {}),
      onStatus: options.onStatus || (() => {}),
      onResult: options.onResult || (() => {}),
      onFailure: options.onFailure || (() => {}),
    };

    let generation = 0;
    let timer = null;
    let runId = null;
    let transientErrors = 0;
    let resultRequested = false;

    function clearTimer() {
      if (timer !== null) cancelSchedule(timer);
      timer = null;
    }

    function cancel() {
      generation += 1;
      clearTimer();
      runId = null;
      resultRequested = false;
      transientErrors = 0;
    }

    function fail(error, phase, token) {
      if (token !== generation) return;
      clearTimer();
      callbacks.onFailure(error, { phase, runId });
    }

    function schedulePoll(token) {
      if (token !== generation) return;
      clearTimer();
      timer = schedule(() => poll(token), pollIntervalMs);
    }

    async function poll(token) {
      if (token !== generation || !runId) return;
      timer = null;
      let status;
      try {
        status = await apiClient.getRunStatus(runId);
      } catch (error) {
        if (token !== generation) return;
        transientErrors += 1;
        if (transientErrors < MAX_TRANSIENT_POLL_ERRORS) {
          callbacks.onStatus({ state: "network_retry", message: "Connection interrupted; retrying status check." });
          schedulePoll(token);
        } else {
          fail(error, "poll", token);
        }
        return;
      }
      if (token !== generation) return;
      transientErrors = 0;
      callbacks.onStatus(status);

      if (status.state === "queued" || status.state === "running") {
        schedulePoll(token);
        return;
      }
      if (status.state === "failed") {
        const message = status.error && status.error.message
          ? status.error.message
          : (status.message || "Pipeline 1 could not complete this dataset.");
        fail(new Error(message), "run", token);
        return;
      }
      if (status.state !== "complete" || resultRequested) return;

      resultRequested = true;
      try {
        const result = await apiClient.getPipeline1Result(runId);
        if (token !== generation) return;
        callbacks.onResult(result);
      } catch (error) {
        fail(error, "result", token);
      }
    }

    async function start(file) {
      cancel();
      const token = generation;
      let accepted;
      try {
        accepted = await apiClient.createRun(file);
      } catch (error) {
        fail(error, "create", token);
        return;
      }
      if (token !== generation) return;
      if (!accepted || !accepted.run_id) {
        fail(new Error("The LYNX service did not return a run identifier."), "create", token);
        return;
      }
      runId = accepted.run_id;
      callbacks.onRunAccepted(accepted);
      await poll(token);
    }

    return {
      start,
      cancel,
      getSnapshot: () => ({ generation, runId, polling: timer !== null, resultRequested }),
    };
  }

  return {
    DEFAULT_POLL_INTERVAL_MS,
    MAX_TRANSIENT_POLL_ERRORS,
    eligibilityMessage,
    createPipeline1Controller,
  };
});
