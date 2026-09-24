/** Prediction run lifecycle controller for the real Stage 4A–4D application API. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.LYNXPredictionController = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const DEFAULT_POLL_INTERVAL_MS = 2000;
  const MAX_TRANSIENT_POLL_ERRORS = 3;

  function safePredictionErrorMessage(error) {
    if (!error) return "Prediction could not be completed.";
    const code = error.code || "";
    const message = error.message || "";
    if (code === "task_type_unresolved" || code === "target_unresolved" || /unresolved/i.test(message)) {
      return "LYNX could not determine a reliable prediction type for this target.";
    }
    return message || "Prediction could not be completed.";
  }

  function createPredictionController(options = {}) {
    const apiClient = options.apiClient;
    const schedule = options.schedule || ((callback, delay) => setTimeout(callback, delay));
    const cancelSchedule = options.cancelSchedule || ((id) => clearTimeout(id));
    const pollIntervalMs = options.pollIntervalMs || DEFAULT_POLL_INTERVAL_MS;

    const callbacks = {
      onJobCreated: options.onJobCreated || (() => {}),
      onStatus: options.onStatus || (() => {}),
      onResult: options.onResult || (() => {}),
      onFailure: options.onFailure || (() => {}),
    };

    let generation = 0;
    let timer = null;
    let runId = null;
    let predictionId = null;
    let targetColumn = null;
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
      predictionId = null;
      targetColumn = null;
      resultRequested = false;
      transientErrors = 0;
    }

    function fail(error, phase, token) {
      if (token !== generation) return;
      clearTimer();
      callbacks.onFailure(error, { phase, runId, predictionId, targetColumn });
    }

    function schedulePoll(token) {
      if (token !== generation) return;
      clearTimer();
      timer = schedule(() => poll(token), pollIntervalMs);
    }

    async function poll(token) {
      if (token !== generation || !runId || !predictionId) return;
      timer = null;
      let status;
      try {
        status = await apiClient.getPredictionStatus(runId, predictionId);
      } catch (error) {
        if (token !== generation) return;
        transientErrors += 1;
        if (transientErrors < MAX_TRANSIENT_POLL_ERRORS) {
          callbacks.onStatus({
            state: "network_retry",
            message: "Connection interrupted; retrying status check.",
          });
          schedulePoll(token);
        } else {
          fail(error, "poll", token);
        }
        return;
      }

      if (token !== generation) return;
      transientErrors = 0;

      // Normal product-level status presentation only (never expose internal stage names)
      callbacks.onStatus({
        state: status.state,
        target_column: targetColumn,
        prediction_id: predictionId,
      });

      if (status.state === "queued" || status.state === "running") {
        schedulePoll(token);
        return;
      }

      if (status.state === "failed") {
        const errorMsg = safePredictionErrorMessage(status.error);
        fail(new Error(errorMsg), "prediction", token);
        return;
      }

      if (status.state !== "complete" || resultRequested) return;

      resultRequested = true;
      try {
        const result = await apiClient.getPredictionResult(runId, predictionId);
        if (token !== generation) return;
        const initialRows = await apiClient.getPredictionRows(runId, predictionId, { offset: 0, limit: 50 });
        if (token !== generation) return;
        callbacks.onResult(result, initialRows);
      } catch (error) {
        fail(error, "result", token);
      }
    }

    async function start(activeRunId, target) {
      cancel();
      const token = generation;
      runId = activeRunId;
      targetColumn = target;

      let created;
      try {
        created = await apiClient.createPrediction(activeRunId, target);
      } catch (error) {
        fail(error, "create", token);
        return;
      }

      if (token !== generation) return;
      if (!created || !created.prediction_id) {
        fail(new Error("The LYNX service did not return a prediction identifier."), "create", token);
        return;
      }

      predictionId = created.prediction_id;
      callbacks.onJobCreated(created);
      await poll(token);
    }

    return {
      start,
      cancel,
      getSnapshot: () => ({
        generation,
        runId,
        predictionId,
        targetColumn,
        polling: timer !== null,
        resultRequested,
      }),
    };
  }

  return {
    DEFAULT_POLL_INTERVAL_MS,
    MAX_TRANSIENT_POLL_ERRORS,
    safePredictionErrorMessage,
    createPredictionController,
  };
});

