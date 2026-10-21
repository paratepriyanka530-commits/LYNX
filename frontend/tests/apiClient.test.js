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

test("application metadata uses the dedicated read-only endpoint", async () => {
  let request;
  const result = await api.getApplicationInfo({
    fetchImpl: async (url, options) => {
      request = { url, options };
      return response({ application: { version: "0.1.0" } });
    },
  });
  assert.equal(request.url, "/api/application-info");
  assert.equal(request.options.method, "GET");
  assert.equal(result.application.version, "0.1.0");
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

test("target resolution detects the user-selected target without creating a prediction", async () => {
  let request;
  await api.resolvePredictionTarget("run/1", "target_col", {
    fetchImpl: async (url, options) => {
      request = { url, options };
      return response({ target_column: "target_col", status: "resolved", task_type: "classification" });
    },
  });
  assert.equal(request.url, "/api/runs/run%2F1/prediction-target-resolution");
  assert.equal(request.options.method, "POST");
  assert.deepEqual(JSON.parse(request.options.body), { target_column: "target_col" });
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

test("dataset pages use bounded run-scoped query parameters", async () => {
  let request;
  await api.getDatasetPage("run/1", {
    version: "original", offset: 20, limit: 20, search: "east",
    sort_column: "Sales", sort_direction: "desc",
    filter_column: "Region", filter_value: "East",
  }, {
    fetchImpl: async (url, options) => {
      request = { url, options };
      return response({ version: "original", rows: [] });
    },
  });
  const [path, queryString] = request.url.split("?");
  const query = new URLSearchParams(queryString);
  assert.equal(path, "/api/runs/run%2F1/dataset");
  assert.deepEqual(Object.fromEntries(query), {
    version: "original", offset: "20", limit: "20", search: "east",
    sort_column: "Sales", sort_direction: "desc",
    filter_column: "Region", filter_value: "East",
  });
  assert.equal(request.options.method, "GET");
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

test("automatic visualization limit is passed through the existing endpoint", async () => {
  let request;
  await api.getAutomaticVisualizations("run/id", {
    maxVisualizations: 4,
    fetchImpl: async (url, options) => {
      request = { url, options };
      return response({ mode: "automatic", visualizations: [] });
    },
  });
  assert.equal(request.url, "/api/visualizations/run%2Fid?max_visualizations=4");
  assert.equal(request.options.method, "GET");
});

test("Insights landing uses the dataset-scoped cached endpoint", async () => {
  const calls = [];
  await api.getInsightsLanding("dataset 1", {
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return response({ status: "READY", summary_text: "Summary", suggested_questions: [] });
    },
  });
  assert.equal(calls[0].url, "/api/insights/dataset%201/landing");
  assert.equal(calls[0].options.method, "GET");
});

test("Export availability carries active artifact identifiers", async () => {
  let request;
  await api.getExportAvailability("run/id", {
    prediction_id: "pred/1", insights_conversation_id: "insight 1",
  }, {
    fetchImpl: async (url, options) => {
      request = { url, options };
      return response({ ready: true, sections: [] });
    },
  });
  assert.equal(request.url, "/api/runs/run%2Fid/exports/availability?prediction_id=pred%2F1&insights_conversation_id=insight+1");
  assert.equal(request.options.method, "GET");
});

test("Dataset and report exports use real download responses", async () => {
  const calls = [];
  const downloadResponse = {
    ok: true, status: 200,
    headers: { get: () => 'attachment; filename="lynx-report.pdf"' },
    blob: async () => new Blob(["payload"]),
  };
  const fetchImpl = async (url, options) => { calls.push({ url, options }); return downloadResponse; };
  const dataset = await api.downloadCleanedDataset("run/id", "xlsx", { fetchImpl });
  const report = await api.generateAnalysisReport("run/id", { format: "pdf", sections: ["overview"] }, { fetchImpl });
  assert.equal(calls[0].url, "/api/runs/run%2Fid/exports/dataset?format=xlsx");
  assert.equal(calls[1].url, "/api/runs/run%2Fid/exports/report");
  assert.deepEqual(JSON.parse(calls[1].options.body), { format: "pdf", sections: ["overview"] });
  assert.equal(dataset.filename, "lynx-report.pdf");
  assert.equal(report.blob.size, 7);
});
