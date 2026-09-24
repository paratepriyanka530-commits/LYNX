const test = require("node:test");
const assert = require("node:assert/strict");
const {
  createPredictionController,
  safePredictionErrorMessage,
} = require("../predictionController.js");

function scheduler() {
  const pending = [];
  return {
    schedule(callback) { pending.push(callback); return callback; },
    cancel(callback) { const index = pending.indexOf(callback); if (index >= 0) pending.splice(index, 1); },
    async next() { const callback = pending.shift(); assert.ok(callback, "expected a scheduled poll"); await callback(); },
    get size() { return pending.length; },
  };
}

function build(apiClient, overrides = {}) {
  const clock = scheduler();
  const events = { created: [], statuses: [], results: [], failures: [] };
  const controller = createPredictionController({
    apiClient,
    schedule: clock.schedule,
    cancelSchedule: clock.cancel,
    onJobCreated: (value) => events.created.push(value),
    onStatus: (value) => events.statuses.push(value),
    onResult: (result, rows) => events.results.push({ result, rows }),
    onFailure: (error, context) => events.failures.push({ error, context }),
    ...overrides,
  });
  return { controller, clock, events };
}

test("prediction creation 202 stores prediction_id and starts polling", async () => {
  const fixture = build({
    createPrediction: async () => ({ run_id: "r1", prediction_id: "p1", state: "queued" }),
    getPredictionStatus: async () => ({ state: "queued" }),
  });
  await fixture.controller.start("r1", "target_col");
  assert.equal(fixture.events.created[0].prediction_id, "p1");
  assert.equal(fixture.controller.getSnapshot().predictionId, "p1");
  assert.equal(fixture.clock.size, 1);
});

test("complete state stops polling and retrieves safe result and initial rows", async () => {
  const fixture = build({
    createPrediction: async () => ({ run_id: "r1", prediction_id: "p1", state: "queued" }),
    getPredictionStatus: async () => ({ state: "complete" }),
    getPredictionResult: async () => ({ status: "complete", task_type: "classification", target: "target_col" }),
    getPredictionRows: async () => ({ offset: 0, limit: 50, total: 10, rows: [] }),
  });
  await fixture.controller.start("r1", "target_col");
  assert.equal(fixture.events.results.length, 1);
  assert.equal(fixture.events.results[0].result.task_type, "classification");
  assert.equal(fixture.clock.size, 0);
});

test("failed state stops polling with controlled failure", async () => {
  const fixture = build({
    createPrediction: async () => ({ run_id: "r1", prediction_id: "p1", state: "queued" }),
    getPredictionStatus: async () => ({ state: "failed", error: { code: "task_type_unresolved" } }),
  });
  await fixture.controller.start("r1", "target_col");
  assert.equal(fixture.events.failures.length, 1);
  assert.equal(
    fixture.events.failures[0].error.message,
    "LYNX could not determine a reliable prediction type for this target."
  );
  assert.equal(fixture.clock.size, 0);
});

test("stale response guard prevents older prediction from overwriting newer run", async () => {
  let slowStatusResolve;
  const slowStatusPromise = new Promise((resolve) => { slowStatusResolve = resolve; });

  const fixture = build({
    createPrediction: async (_runId, target) => ({
      run_id: "r1",
      prediction_id: `pred_${target}`,
      state: "queued",
    }),
    getPredictionStatus: async (_runId, predId) => {
      if (predId === "pred_first") {
        await slowStatusPromise;
        return { state: "complete" };
      }
      return { state: "running" };
    },
    getPredictionResult: async () => ({ status: "complete", target: "first" }),
    getPredictionRows: async () => ({ offset: 0, limit: 50, total: 0, rows: [] }),
  });

  // Start first job
  const firstPromise = fixture.controller.start("r1", "first");
  // Immediately start second job before first status completes
  await fixture.controller.start("r1", "second");
  assert.equal(fixture.controller.getSnapshot().targetColumn, "second");

  // Let first slow status resolve now
  slowStatusResolve();
  await firstPromise;

  // Verify first result was discarded and did not fire onResult
  assert.equal(fixture.events.results.length, 0);
});

test("transient errors retry up to limit then fail gracefully", async () => {
  let callCount = 0;
  const fixture = build({
    createPrediction: async () => ({ run_id: "r1", prediction_id: "p1", state: "queued" }),
    getPredictionStatus: async () => {
      callCount++;
      throw new Error("Network blip");
    },
  });
  await fixture.controller.start("r1", "target_col");
  assert.equal(fixture.events.statuses[0].state, "network_retry");
  assert.equal(fixture.clock.size, 1);

  await fixture.clock.next(); // 2nd try
  assert.equal(fixture.clock.size, 1);

  await fixture.clock.next(); // 3rd try -> should fail
  assert.equal(fixture.events.failures.length, 1);
  assert.equal(fixture.events.failures[0].context.phase, "poll");
  assert.equal(fixture.clock.size, 0);
});

