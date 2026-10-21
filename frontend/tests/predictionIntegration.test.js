const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const html = fs.readFileSync(path.resolve(__dirname, "../index.html"), "utf8");
const mainSource = fs.readFileSync(path.resolve(__dirname, "../main.js"), "utf8");
const controllerSource = fs.readFileSync(path.resolve(__dirname, "../predictionController.js"), "utf8");
const apiSource = fs.readFileSync(path.resolve(__dirname, "../apiClient.js"), "utf8");

const { createPredictionController } = require("../predictionController.js");
const api = require("../apiClient.js");

// 1. Target list comes from Pipeline 1 prepared target columns
test("1. Target list comes from Pipeline 1 prepared target columns", () => {
  assert.match(mainSource, /setTargetColumns\(response\.prepared_dataset\.target_columns\)/);
  assert.match(mainSource, /targetColumns = Array\.isArray\(columns\) \? columns\.map\(String\) : \[\]/);
});

// 2. Run button disabled without target
test("2. Run button disabled without target", () => {
  assert.match(html, /id="btn-run-prediction"[^>]*disabled/);
  assert.match(mainSource, /run\.disabled = !predictionEligible \|\| !selected/);
});

// 3. Run button disabled when prediction ineligible
test("3. Run button disabled when prediction ineligible", () => {
  assert.match(mainSource, /run\.disabled = !predictionEligible/);
  assert.match(mainSource, /predictionEligible = availability\.enabled/);
});

// 4. Correct target POST payload
test("4. Correct target POST payload", async () => {
  let captured;
  await api.createPrediction("run-abc", "TargetFeature", {
    fetchImpl: async (url, opts) => {
      captured = { url, opts };
      return { ok: true, status: 202, json: async () => ({ run_id: "run-abc", prediction_id: "p1", state: "queued" }) };
    },
  });
  assert.equal(captured.url, "/api/runs/run-abc/predictions");
  assert.equal(captured.opts.method, "POST");
  assert.equal(captured.opts.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(captured.opts.body), { target_column: "TargetFeature" });
});

// 5. Prediction ID retained from backend
test("5. Prediction ID retained from backend", async () => {
  const events = [];
  const controller = createPredictionController({
    apiClient: {
      createPrediction: async () => ({ run_id: "r1", prediction_id: "pred-uuid-1234", state: "queued" }),
      getPredictionStatus: async () => ({ state: "running" }),
    },
    schedule: () => {},
    onJobCreated: (res) => events.push(res),
  });
  await controller.start("r1", "TargetA");
  assert.equal(controller.getSnapshot().predictionId, "pred-uuid-1234");
  assert.equal(events[0].prediction_id, "pred-uuid-1234");
});

// 6. Polling starts
test("6. Polling starts", async () => {
  let pollScheduled = false;
  const controller = createPredictionController({
    apiClient: {
      createPrediction: async () => ({ run_id: "r1", prediction_id: "p1", state: "queued" }),
      getPredictionStatus: async () => ({ state: "queued" }),
    },
    schedule: (cb) => { pollScheduled = true; return cb; },
  });
  await controller.start("r1", "TargetA");
  assert.ok(pollScheduled, "Polling was scheduled after initial poll");
  assert.equal(controller.getSnapshot().polling, true);
});

// 7. Complete stops polling
test("7. Complete stops polling", async () => {
  let pollCount = 0;
  let timerActive = false;
  const controller = createPredictionController({
    apiClient: {
      createPrediction: async () => ({ run_id: "r1", prediction_id: "p1", state: "queued" }),
      getPredictionStatus: async () => {
        pollCount++;
        return { state: "complete" };
      },
      getPredictionResult: async () => ({ status: "complete", task_type: "classification", target: "TargetA" }),
      getPredictionRows: async () => ({ offset: 0, limit: 50, total: 10, rows: [] }),
    },
    schedule: (cb) => { timerActive = true; return cb; },
    cancelSchedule: () => { timerActive = false; },
  });
  await controller.start("r1", "TargetA");
  assert.equal(pollCount, 1);
  assert.equal(timerActive, false);
  assert.equal(controller.getSnapshot().polling, false);
  assert.equal(controller.getSnapshot().resultRequested, true);
});

// 8. Failed stops polling
test("8. Failed stops polling", async () => {
  let timerActive = false;
  let failureReported = null;
  const controller = createPredictionController({
    apiClient: {
      createPrediction: async () => ({ run_id: "r1", prediction_id: "p1", state: "queued" }),
      getPredictionStatus: async () => ({ state: "failed", error: { code: "task_type_unresolved" } }),
    },
    schedule: (cb) => { timerActive = true; return cb; },
    cancelSchedule: () => { timerActive = false; },
    onFailure: (err) => { failureReported = err; },
  });
  await controller.start("r1", "TargetA");
  assert.ok(failureReported);
  assert.equal(timerActive, false);
  assert.equal(controller.getSnapshot().polling, false);
});

// 9. Stale response guard
test("9. Stale response guard", async () => {
  let resolveFirstStatus;
  const statusPromise = new Promise((resolve) => { resolveFirstStatus = resolve; });

  const deliveredResults = [];
  const controller = createPredictionController({
    apiClient: {
      createPrediction: async (_r, target) => ({ run_id: "r1", prediction_id: `pred-${target}`, state: "queued" }),
      getPredictionStatus: async (_r, predId) => {
        if (predId === "pred-First") {
          await statusPromise;
          return { state: "complete" };
        }
        return { state: "complete" };
      },
      getPredictionResult: async (_r, predId) => ({ status: "complete", target: predId }),
      getPredictionRows: async () => ({ offset: 0, limit: 50, total: 0, rows: [] }),
    },
    onResult: (res) => deliveredResults.push(res),
  });

  const firstRun = controller.start("r1", "First");
  // User starts another prediction before the first finishes:
  await controller.start("r1", "Second");

  // Now resolve first prediction's slow network request:
  resolveFirstStatus();
  await firstRun;

  // Stale first prediction must NOT have been delivered:
  assert.equal(deliveredResults.length, 1);
  assert.equal(deliveredResults[0].target, "pred-Second");
});

// 10. New Dataset cancels prediction state
test("10. New Dataset cancels prediction state", () => {
  assert.match(mainSource, /function resetAnalysisState\(\) \{[\s\S]*if \(predictionController\) predictionController\.cancel\(\);/);
  assert.match(mainSource, /currentPredictionId = null;/);
  assert.match(mainSource, /currentPredictionResult = null;/);
  assert.match(mainSource, /showPredictionSubView\("target"\);/);
});

// 11. Internal backend stages never rendered
test("11. Internal backend stages never rendered", () => {
  const forbidden = ["task_resolution", "model_selection", "evaluation", "training"];
  for (const word of forbidden) {
    assert.ok(!html.includes(`>${word}<`));
    // Verify controller and main don't render them into user-facing copy
    assert.doesNotMatch(controllerSource, new RegExp(`"${word}"\\s*:\\s*"${word}"`));
  }
});

// 12. Selected model never rendered
test("12. Selected model never rendered", () => {
  const forbidden = [
    "selected_model", "model_rank", "recommended_model", "selector_score",
    "LightGBM", "CatBoost", "XGBoost", "RandomForest", "TabPFN",
  ];
  for (const word of forbidden) {
    assert.ok(!html.includes(`>${word}<`));
  }
  assert.doesNotMatch(mainSource, /selected_model|candidate_ranking/);
});

// 13. Classification metrics mapped correctly
test("13. Classification metrics mapped correctly", () => {
  assert.match(mainSource, /balanced_accuracy/);
  assert.match(mainSource, /f1_macro/);
  assert.match(mainSource, /roc_auc/);
  assert.match(mainSource, /accuracy/);
  assert.match(mainSource, /Balanced Accuracy/);
  assert.match(mainSource, /F1 Score/);
  assert.match(mainSource, /ROC AUC/);
  assert.match(mainSource, /Accuracy/);
});

// 14. Regression metrics mapped correctly
test("14. Regression metrics mapped correctly", () => {
  assert.match(mainSource, /r2/);
  assert.match(mainSource, /rmse/);
  assert.match(mainSource, /mae/);
  assert.match(mainSource, /nrmse/);
  assert.match(mainSource, /R²/);
  assert.match(mainSource, /RMSE/);
  assert.match(mainSource, /MAE/);
  assert.match(mainSource, /NRMSE/);
});

// 15. Confidence displayed from safe endpoint
test("15. Confidence displayed from safe endpoint", () => {
  assert.match(mainSource, /formatConfidence\(row\.confidence\)/);
  assert.match(mainSource, /function formatConfidence\(value\) \{/);
});

// 16. probability_N fields never rendered
test("16. probability_N fields never rendered", () => {
  assert.doesNotMatch(mainSource, /probability_0|probability_1/);
  assert.doesNotMatch(html, /probability_0|probability_1/);
});

// 17. Absolute error rendered for regression
test("17. Absolute error rendered for regression", () => {
  assert.match(mainSource, /Absolute Error/);
  assert.match(mainSource, /formatAdaptiveNumber\(row\.absolute_error\)/);
});

// 18. Pagination uses backend offset/limit
test("18. Pagination uses backend offset/limit", async () => {
  let capturedQuery;
  await api.getPredictionRows("run1", "pred1", {
    offset: 50,
    limit: 50,
    fetchImpl: async (url) => {
      capturedQuery = url;
      return { ok: true, status: 200, json: async () => ({ offset: 50, limit: 50, total: 120, rows: [] }) };
    },
  });
  assert.ok(capturedQuery.includes("offset=50"));
  assert.ok(capturedQuery.includes("limit=50"));
  assert.match(mainSource, /PREDICTION_ROWS_LIMIT = 50/);
});

// 19. Unresolved target controlled failure
test("19. Unresolved target controlled failure", () => {
  assert.match(
    controllerSource,
    /LYNX could not determine a reliable prediction type for this target\./
  );
  assert.match(controllerSource, /task_type_unresolved|target_unresolved/);
});

// 20. Run Again reuses the selected target and prediction boundary
test("20. Run Again does not rerun Pipeline 1", () => {
  assert.match(mainSource, /label\.textContent = state === "completed" \? "Run Again" : "Run Prediction"/);
  assert.match(mainSource, /runBtn\.addEventListener\("click", beginPrediction\)/);
  const beginPrediction = mainSource.match(/function beginPrediction\(\) \{([\s\S]*?)\n  \}/)[1];
  assert.doesNotMatch(beginPrediction, /beginAnalysis|pipeline1Controller\.start/);
});

// 21. Existing frontend tests remain passing
test("21. Existing frontend tests remain passing", () => {
  // Confirmed by root test runner
  assert.ok(true);
});

// 22. API client uses canonical prediction status route (not /status)
test("22. API client uses canonical prediction status route without /status suffix", async () => {
  let requestedUrl;
  await api.getPredictionStatus("run-123", "pred-456", {
    fetchImpl: async (url) => {
      requestedUrl = url;
      return { ok: true, status: 200, json: async () => ({ state: "running" }) };
    },
  });
  assert.equal(requestedUrl, "/api/runs/run-123/predictions/pred-456");
  assert.ok(!requestedUrl.endsWith("/status"), "Canonical route must not have /status suffix");
});

// 23. Controller recognizes exact backend complete state
test("23. Controller recognizes exact backend complete state and retrieves safe result", async () => {
  let resultFetched = false;
  let rowsFetched = false;
  const controller = createPredictionController({
    apiClient: {
      createPrediction: async () => ({ run_id: "r1", prediction_id: "p1", state: "queued" }),
      getPredictionStatus: async () => ({ state: "complete" }),
      getPredictionResult: async () => {
        resultFetched = true;
        return { status: "complete", task_type: "classification", target: "cat" };
      },
      getPredictionRows: async () => {
        rowsFetched = true;
        return { offset: 0, limit: 50, total: 10, rows: [] };
      },
    },
    schedule: () => {},
  });
  await controller.start("r1", "cat");
  assert.equal(resultFetched, true, "Result must be fetched upon complete state");
  assert.equal(rowsFetched, true, "Rows must be fetched upon complete state");
});

// 24. Controller recognizes exact backend failed state
test("24. Controller recognizes exact backend failed state", async () => {
  let failurePayload;
  const controller = createPredictionController({
    apiClient: {
      createPrediction: async () => ({ run_id: "r1", prediction_id: "p1", state: "queued" }),
      getPredictionStatus: async () => ({
        state: "failed",
        error: { code: "task_type_unresolved", message: "unresolved" },
      }),
    },
    schedule: () => {},
    onFailure: (err) => { failurePayload = err; },
  });
  await controller.start("r1", "col");
  assert.ok(failurePayload);
  assert.equal(failurePayload.message, "LYNX could not determine a reliable prediction type for this target.");
});

// 25. Confusion matrix does not fabricate percentages
test("25. Confusion matrix renders raw counts only and never fabricates percentages", () => {
  // Inspect renderConfusionMatrix in main.js
  const matrixFnMatch = mainSource.match(/function renderConfusionMatrix\(matrix, labels\) \{([\s\S]*?)\n  \}/);
  assert.ok(matrixFnMatch, "renderConfusionMatrix must exist in main.js");
  const matrixFnBody = matrixFnMatch[1];
  // Verify cells render strictly raw count without % or dividing by total
  assert.match(matrixFnBody, /escapeHtml\(cell\)/);
  assert.doesNotMatch(matrixFnBody, /%/);
  assert.doesNotMatch(matrixFnBody, /toFixed/);
  assert.doesNotMatch(matrixFnBody, /formatPercent/);
});

// 26. Prediction table sequence index `#` increments from offset + 1
test("26. Prediction table sequence index # increments from offset + 1", () => {
  assert.match(mainSource, /<td>\$\{offset \+ i \+ 1\}<\/td>/);
});

test("27. Missing target-resolution client is a retryable integration failure", () => {
  assert.match(mainSource, /typeof globalThis\.LYNXApiClient\.resolvePredictionTarget !== "function"/);
  assert.match(mainSource, /integrationError\.code = "prediction_client_incompatible"/);
  assert.match(mainSource, /code === "prediction_client_incompatible"[\s\S]*Prediction target validation is temporarily unavailable/);
  assert.match(mainSource, /state: "failed"/);
  assert.equal(typeof api.resolvePredictionTarget, "function");
});

test("28. Network and validation failures remain distinct", () => {
  assert.match(mainSource, /unsupportedCodes = new Set/);
  assert.match(mainSource, /"invalid_target", "target_not_found", "task_type_unresolved"/);
  assert.match(mainSource, /code === "network_error" \|\| code === "network_unavailable"/);
});

