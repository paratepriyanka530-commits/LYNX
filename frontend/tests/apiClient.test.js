const test = require("node:test");
const assert = require("node:assert/strict");
const { File } = require("node:buffer");
const api = require("../apiClient.js");

function response(body, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

test("createRun sends the actual File under the exact dataset multipart key", async () => {
  const file = new File(["a,b\n1,2\n"], "sample.csv", { type: "text/csv" });
  let request;
  const result = await api.createRun(file, {
    fetchImpl: async (url, options) => {
      request = { url, options };
      return response({ run_id: "run-1", state: "queued" }, 202);
    },
  });
  assert.equal(request.url, "/api/runs");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.body.get("dataset"), file);
  assert.deepEqual([...request.options.body.keys()], ["dataset"]);
  assert.equal(result.run_id, "run-1");
});

test("status and result calls use encoded same-origin API paths", async () => {
  const calls = [];
  const fetchImpl = async (url) => {
    calls.push(url);
    return response({ state: "complete" });
  };
  await api.getRunStatus("run/id", { fetchImpl });
  await api.getPipeline1Result("run/id", { fetchImpl });
  assert.deepEqual(calls, ["/api/runs/run%2Fid", "/api/runs/run%2Fid/pipeline1"]);
});

test("controlled backend errors expose only the safe message", async () => {
  await assert.rejects(
    api.getRunStatus("bad", {
      fetchImpl: async () => response({ error: { code: "not_found", message: "Run not found." } }, 404),
    }),
    (error) => error.code === "not_found" && error.message === "Run not found." && error.status === 404
  );
});

test("non-JSON errors are normalized without exposing raw HTML", async () => {
  await assert.rejects(
    api.getRunStatus("bad", {
      fetchImpl: async () => ({ ok: false, status: 502, json: async () => { throw new Error("<html>secret</html>"); } }),
    }),
    (error) => error.message === "The LYNX service rejected the request (HTTP 502)." && !error.message.includes("html")
  );
});

test("network failures receive a controlled user-facing error", async () => {
  await assert.rejects(
    api.getRunStatus("run", { fetchImpl: async () => { throw new Error("socket internals"); } }),
    (error) => error.code === "network_error" && !error.message.includes("socket internals")
  );
});

test("createPrediction sends correct target_column JSON body and headers", async () => {
  let request;
  const result = await api.createPrediction("run-123", "target_col", {
    fetchImpl: async (url, options) => {
      request = { url, options };
      return response({ run_id: "run-123", prediction_id: "pred-456", state: "queued" }, 202);
    },
  });
  assert.equal(request.url, "/api/runs/run-123/predictions");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(request.options.body), { target_column: "target_col" });
  assert.equal(result.prediction_id, "pred-456");
});

test("prediction status, result, and rows calls use encoded same-origin paths and query params", async () => {
  const calls = [];
  const fetchImpl = async (url) => {
    calls.push(url);
    return response({ state: "complete", rows: [] });
  };
  await api.getPredictionStatus("run/1", "pred/1", { fetchImpl });
  await api.getPredictionResult("run/1", "pred/1", { fetchImpl });
  await api.getPredictionRows("run/1", "pred/1", { offset: 10, limit: 25, fetchImpl });
  assert.deepEqual(calls, [
    "/api/runs/run%2F1/predictions/pred%2F1",
    "/api/runs/run%2F1/predictions/pred%2F1/result",
    "/api/runs/run%2F1/predictions/pred%2F1/rows?offset=10&limit=25",
  ]);
});

test("createConversation sends the exact first-turn payload", async () => {
  let request;
  await api.createConversation("run/id", "  show sales by region  ", {
    fetchImpl: async (url, options) => {
      request = { url, options };
      return response({ conversation_id: "conv-1", turn_id: "turn-1" });
    },
  });
  assert.equal(request.url, "/api/conversations/run%2Fid");
  assert.equal(request.options.method, "POST");
  assert.deepEqual(JSON.parse(request.options.body), { message: "show sales by region" });
});

test("sendConversationMessage uses the dataset-scoped continuation endpoint", async () => {
  let request;
  await api.sendConversationMessage("dataset 1", "conversation/1", "use total instead", {
    fetchImpl: async (url, options) => {
      request = { url, options };
      return response({ conversation_id: "conversation/1", turn_id: "turn-2" });
    },
  });
  assert.equal(request.url, "/api/conversations/dataset%201/conversation%2F1");
  assert.deepEqual(JSON.parse(request.options.body), { message: "use total instead" });
});

test("automatic visualizations use an independent GET endpoint", async () => {
  let request;
  await api.getAutomaticVisualizations("run/id", {
    fetchImpl: async (url, options) => {
      request = { url, options };
      return response({ mode: "automatic", visualizations: [] });
    },
  });
  assert.equal(request.url, "/api/visualizations/run%2Fid");
  assert.equal(request.options.method, "GET");
});
