const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { createConversationController } = require("../conversationController.js");
const charts = require("../chartRenderer.js");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const main = fs.readFileSync(path.join(root, "main.js"), "utf8");

function result(overrides = {}) {
  return {
    conversation_id: "conversation-1", turn_id: "turn-1", intent: "INSIGHT",
    status: "READY", message: "Complete.", visualization: null,
    insight: { status: "READY", answer: "Engineering is highest." },
    clarification_options: [], ...overrides,
  };
}

test("Insights creates a new dataset-scoped conversation", async () => {
  const calls = [];
  const controller = createConversationController({ apiClient: {
    createConversation: async (...args) => { calls.push(args); return result(); },
    sendConversationMessage: async () => { throw new Error("unexpected"); },
  } });
  controller.setDataset("run-1");
  await controller.submit("Which department is highest?");
  assert.deepEqual(calls[0], ["run-1", "Which department is highest?"]);
  assert.equal(controller.getState().messages.length, 2);
});

test("Visualization creates a new conversation and retains chart payload", async () => {
  const visualization = { status: "READY", visualizations: [{ chart_type: "BAR", chart_data: { points: [{ x: "A", y: 2 }] } }] };
  const controller = createConversationController({ apiClient: {
    createConversation: async () => result({ intent: "VISUALIZATION", visualization, insight: null }),
  } });
  controller.setDataset("run-1");
  await controller.submit("Show sales by region");
  assert.deepEqual(controller.getState().messages[1].visualization, visualization);
});

test("conversation_id is stored and follow-up uses it without reconstruction", async () => {
  const calls = [];
  const controller = createConversationController({ apiClient: {
    createConversation: async () => result(),
    sendConversationMessage: async (...args) => { calls.push(args); return result({ turn_id: "turn-2" }); },
  } });
  controller.setDataset("run-1");
  await controller.submit("First question");
  assert.equal(controller.getState().conversationId, "conversation-1");
  await controller.submit("What about the lowest?");
  assert.deepEqual(calls[0], ["run-1", "conversation-1", "What about the lowest?"]);
});

test("active dataset change clears conversation id, messages, and errors", async () => {
  const controller = createConversationController({ apiClient: { createConversation: async () => result() } });
  controller.setDataset("run-1");
  await controller.submit("Question");
  controller.setDataset("run-2");
  assert.deepEqual(controller.getState(), { datasetId: "run-2", conversationId: null, messages: [], loading: false, error: null });
});

test("same dataset does not reset an active conversation", async () => {
  const controller = createConversationController({ apiClient: { createConversation: async () => result() } });
  controller.setDataset("run-1");
  await controller.submit("Question");
  controller.setDataset("run-1");
  assert.equal(controller.getState().conversationId, "conversation-1");
});

test("READY Insight stores the backend answer and evidence unchanged", async () => {
  const insight = { status: "READY", answer: "Total revenue is 42.", evidence: { total_rows: 10, usable_rows: 9 } };
  const controller = createConversationController({ apiClient: { createConversation: async () => result({ insight }) } });
  controller.setDataset("run-1");
  await controller.submit("Total revenue?");
  assert.deepEqual(controller.getState().messages[1].insight, insight);
});

test("NEEDS_CLARIFICATION stores clickable choice data", async () => {
  const controller = createConversationController({ apiClient: { createConversation: async () => result({ status: "NEEDS_CLARIFICATION", clarification_options: ["Gross Sales", "Net Sales"] }) } });
  controller.setDataset("run-1");
  await controller.submit("Show sales");
  assert.deepEqual(controller.getState().messages[1].clarificationOptions, ["Gross Sales", "Net Sales"]);
  assert.match(main, /data-conversation-choice/);
  assert.match(main, /conversation\.submit\(choice\.dataset\.conversationChoice/);
});

test("INVALID is a controlled assistant turn rather than a service error", async () => {
  const controller = createConversationController({ apiClient: { createConversation: async () => result({ status: "INVALID", message: "No rainfall column was found.", insight: null }) } });
  controller.setDataset("run-1");
  await controller.submit("Show rainfall");
  assert.equal(controller.getState().messages[1].status, "INVALID");
  assert.equal(controller.getState().error, null);
});

test("network error is kept separate from analytical responses", async () => {
  const error = Object.assign(new Error("private socket detail"), { code: "network_error" });
  const controller = createConversationController({ apiClient: { createConversation: async () => { throw error; } } });
  controller.setDataset("run-1");
  await controller.submit("Question");
  assert.equal(controller.getState().messages.length, 1);
  assert.equal(controller.getState().error.code, "network_error");
  assert.match(main, /Unable to process this request right now/);
});

test("cross-intent Insight and Visualization payloads are retained by response intent", async () => {
  let turn = 0;
  const controller = createConversationController({ apiClient: {
    createConversation: async () => result({ intent: "INSIGHT" }),
    sendConversationMessage: async () => result({ turn_id: `turn-${++turn + 1}`, intent: "VISUALIZATION", insight: null, visualization: { status: "READY", visualizations: [] } }),
  } });
  controller.setDataset("run-1");
  await controller.submit("Which is highest?");
  await controller.submit("Show that as a chart");
  assert.deepEqual(controller.getState().messages.filter((m) => m.role === "assistant").map((m) => m.intent), ["INSIGHT", "VISUALIZATION"]);
  assert.match(main, /message\.intent === "VISUALIZATION"/);
  assert.match(main, /message\.intent === "INSIGHT"/);
});

test("loading state is observable and blocks duplicate submissions", async () => {
  let resolve;
  let calls = 0;
  const controller = createConversationController({ apiClient: { createConversation: async () => { calls += 1; return new Promise((done) => { resolve = done; }); } } });
  controller.setDataset("run-1");
  const pending = controller.submit("Question");
  assert.equal(controller.getState().loading, true);
  assert.equal(await controller.submit("Duplicate"), null);
  assert.equal(calls, 1);
  resolve(result());
  await pending;
  assert.equal(controller.getState().loading, false);
});

test("renderer supports every Pipeline 3 chart type", () => {
  for (const type of ["BAR", "COUNT_BAR", "LINE", "SCATTER"]) {
    const markup = charts.visualizationMarkup({ chart_type: type, chart_data: { chart_type: type, x_column: "x", y_column: "y", points: [{ x: 1, y: 2 }], has_data: true } });
    assert.match(markup, new RegExp(`data-chart-type="${type}"`));
    assert.match(markup, /role="img"/);
  }
  const histogram = charts.visualizationMarkup({ chart_type: "HISTOGRAM", chart_data: { chart_type: "HISTOGRAM", x_column: "x", bins: [{ bin_start: 0, bin_end: 1, count: 3 }], has_data: true } });
  assert.match(histogram, /data-chart-type="HISTOGRAM"/);
});

test("automatic recommendations and Ask LYNX have independent containers", () => {
  assert.match(html, /id="visualization-grid"/);
  assert.match(html, /id="visualization-messages"/);
  assert.match(main, /getAutomaticVisualizations\(requestedDataset\)/);
  assert.doesNotMatch(main, /visualization-messages[\s\S]{0,100}innerHTML = visualizations/);
});

test("composer keyboard contract sends on Enter and preserves Shift+Enter", () => {
  assert.match(main, /event\.key === "Enter" && !event\.shiftKey/);
  assert.match(main, /form\.requestSubmit\(\)/);
  assert.match(html, /<textarea id="insights-input"/);
  assert.match(html, /<textarea id="visualization-input"/);
});

test("conversation state is not persisted in browser storage", () => {
  const controllerSource = fs.readFileSync(path.join(root, "conversationController.js"), "utf8");
  assert.doesNotMatch(controllerSource, /localStorage|sessionStorage/);
});
