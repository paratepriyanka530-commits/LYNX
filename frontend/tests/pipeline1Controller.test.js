const test = require("node:test");
const assert = require("node:assert/strict");
const {
  createPipeline1Controller,
  eligibilityMessage,
} = require("../pipeline1Controller.js");

function scheduler() {
  const pending = [];
  return {
    schedule(callback) { pending.push(callback); return callback; },
    cancel(callback) { const index = pending.indexOf(callback); if (index >= 0) pending.splice(index, 1); },
    async next() { const callback = pending.shift(); assert.ok(callback, "expected a scheduled poll"); await callback(); },
    get size() { return pending.length; },
  };
}

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

function build(apiClient, overrides = {}) {
  const clock = scheduler();
  const events = { accepted: [], statuses: [], results: [], failures: [] };
  const controller = createPipeline1Controller({
    apiClient,
    schedule: clock.schedule,
    cancelSchedule: clock.cancel,
    onRunAccepted: (value) => events.accepted.push(value),
    onStatus: (value) => events.statuses.push(value),
    onResult: (value) => events.results.push(value),
    onFailure: (error, context) => events.failures.push({ error, context }),
    ...overrides,
  });
  return { controller, clock, events };
}

test("a successful 202 stores run_id and exposes the processing transition hook", async () => {
  const fixture = build({
    createRun: async () => ({ run_id: "r1", state: "queued" }),
    getRunStatus: async () => ({ state: "queued", message: "Queued" }),
  });
  await fixture.controller.start({ name: "real.csv" });
  assert.equal(fixture.events.accepted[0].run_id, "r1");
  assert.equal(fixture.controller.getSnapshot().runId, "r1");
  assert.equal(fixture.clock.size, 1);
});

test("upload rejection is reported in the create phase without polling", async () => {
  const fixture = build({
    createRun: async () => { throw new Error("Dataset exceeds the upload limit."); },
    getRunStatus: async () => { throw new Error("must not poll"); },
  });
  await fixture.controller.start({ name: "large.csv" });
  assert.equal(fixture.events.accepted.length, 0);
  assert.equal(fixture.events.failures[0].context.phase, "create");
  assert.equal(fixture.clock.size, 0);
});

test("polling reports queued then running without fabricating a percentage", async () => {
  const states = [
    { state: "queued", message: "Waiting" },
    { state: "running", stage: "pipeline1", message: "Analyzing" },
  ];
  const fixture = build({
    createRun: async () => ({ run_id: "r1" }),
    getRunStatus: async () => states.shift(),
  });
  await fixture.controller.start({});
  await fixture.clock.next();
  assert.deepEqual(fixture.events.statuses.map((item) => item.state), ["queued", "running"]);
  assert.ok(fixture.events.statuses.every((item) => !JSON.stringify(item).includes("%")));
  assert.equal(fixture.clock.size, 1);
});

test("complete fetches the Pipeline 1 result exactly once and stops polling", async () => {
  let resultCalls = 0;
  const result = { pipeline1_report: { status: "success" }, prepared_dataset: { target_columns: ["target"] } };
  const fixture = build({
    createRun: async () => ({ run_id: "r1" }),
    getRunStatus: async () => ({ state: "complete" }),
    getPipeline1Result: async () => { resultCalls += 1; return result; },
  });
  await fixture.controller.start({});
  assert.equal(resultCalls, 1);
  assert.equal(fixture.events.results[0], result);
  assert.equal(fixture.clock.size, 0);
  assert.equal(fixture.controller.getSnapshot().resultRequested, true);
});

test("result retrieval error is controlled and does not restart polling", async () => {
  const fixture = build({
    createRun: async () => ({ run_id: "r1" }),
    getRunStatus: async () => ({ state: "complete" }),
    getPipeline1Result: async () => { throw new Error("Result unavailable."); },
  });
  await fixture.controller.start({});
  assert.equal(fixture.events.failures[0].context.phase, "result");
  assert.equal(fixture.clock.size, 0);
});

test("failed backend run stops polling and remains a controlled failure", async () => {
  const fixture = build({
    createRun: async () => ({ run_id: "r1" }),
    getRunStatus: async () => ({ state: "failed", error: { message: "Dataset is malformed." } }),
  });
  await fixture.controller.start({});
  assert.equal(fixture.clock.size, 0);
  assert.equal(fixture.events.failures[0].error.message, "Dataset is malformed.");
  assert.equal(fixture.events.failures[0].context.phase, "run");
});

test("transient polling failures retry and eventually recover", async () => {
  let calls = 0;
  const fixture = build({
    createRun: async () => ({ run_id: "r1" }),
    getRunStatus: async () => {
      calls += 1;
      if (calls === 1) throw new Error("temporary");
      return { state: "queued" };
    },
  });
  await fixture.controller.start({});
  assert.equal(fixture.events.statuses[0].state, "network_retry");
  await fixture.clock.next();
  assert.equal(fixture.events.statuses[1].state, "queued");
  assert.equal(fixture.events.failures.length, 0);
});

test("three consecutive polling failures stop with a network failure", async () => {
  const fixture = build({
    createRun: async () => ({ run_id: "r1" }),
    getRunStatus: async () => { throw new Error("offline"); },
  });
  await fixture.controller.start({});
  await fixture.clock.next();
  await fixture.clock.next();
  assert.equal(fixture.events.failures.length, 1);
  assert.equal(fixture.events.failures[0].context.phase, "poll");
  assert.equal(fixture.clock.size, 0);
});

test("new upload cancels the previous scheduled poll", async () => {
  let id = 0;
  const fixture = build({
    createRun: async () => ({ run_id: `r${++id}` }),
    getRunStatus: async () => ({ state: "queued" }),
  });
  await fixture.controller.start({ name: "old.csv" });
  assert.equal(fixture.clock.size, 1);
  await fixture.controller.start({ name: "new.csv" });
  assert.equal(fixture.controller.getSnapshot().runId, "r2");
  assert.equal(fixture.clock.size, 1);
});

test("a stale old-run create response cannot replace the current run", async () => {
  const old = deferred();
  let createCalls = 0;
  const fixture = build({
    createRun: async () => (++createCalls === 1 ? old.promise : { run_id: "new" }),
    getRunStatus: async () => ({ state: "complete" }),
    getPipeline1Result: async (runId) => ({ run_id: runId, pipeline1_report: {} }),
  });
  const oldStart = fixture.controller.start({ name: "old.csv" });
  await fixture.controller.start({ name: "new.csv" });
  old.resolve({ run_id: "old" });
  await oldStart;
  assert.deepEqual(fixture.events.accepted.map((item) => item.run_id), ["new"]);
  assert.deepEqual(fixture.events.results.map((item) => item.run_id), ["new"]);
});

test("eligibility enables prediction only with approval and target columns", () => {
  assert.equal(eligibilityMessage({
    prepared_dataset: { target_columns: ["label"] },
    prediction_eligibility: { eligible: true, reasons: [] },
  }).enabled, true);
  assert.equal(eligibilityMessage({
    prepared_dataset: { target_columns: [] },
    prediction_eligibility: { eligible: true, reasons: [] },
  }).enabled, false);
  assert.equal(eligibilityMessage({
    prepared_dataset: { target_columns: ["label"] },
    prediction_eligibility: { eligible: false, reasons: ["privacy_clearance_failed"] },
  }).enabled, false);
});

test("ineligible results are still delivered for Pipeline 1 rendering", async () => {
  const result = {
    pipeline1_report: { status: "success" },
    prepared_dataset: { target_columns: ["label"] },
    prediction_eligibility: { eligible: false, reasons: ["privacy_clearance_failed"] },
  };
  const fixture = build({
    createRun: async () => ({ run_id: "r1" }),
    getRunStatus: async () => ({ state: "complete" }),
    getPipeline1Result: async () => result,
  });
  await fixture.controller.start({});
  assert.equal(fixture.events.results[0], result);
  assert.equal(eligibilityMessage(fixture.events.results[0]).enabled, false);
});
