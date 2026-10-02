/** Dataset-scoped conversation state for the Insights and Visualization surfaces. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.LYNXConversationController = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function createConversationController({ apiClient, onChange = () => {}, now = () => new Date().toISOString() }) {
    if (!apiClient) throw new TypeError("apiClient is required");
    let datasetId = null;
    let conversationId = null;
    let messages = [];
    let loading = false;
    let error = null;
    let sequence = 0;

    function snapshot() {
      return { datasetId, conversationId, messages: messages.map((item) => ({ ...item })), loading, error };
    }

    function emit() { onChange(snapshot()); }

    function reset(nextDatasetId = null) {
      datasetId = nextDatasetId || null;
      conversationId = null;
      messages = [];
      loading = false;
      error = null;
      emit();
    }

    function setDataset(nextDatasetId) {
      const normalized = nextDatasetId ? String(nextDatasetId) : null;
      if (normalized !== datasetId) reset(normalized);
    }

    async function submit(rawText) {
      const text = String(rawText || "").trim();
      if (!text || loading) return null;
      if (!datasetId) throw new Error("A dataset is required before starting a conversation.");

      messages.push({ id: `local-${++sequence}`, role: "user", text, createdAt: now() });
      loading = true;
      error = null;
      emit();

      try {
        const response = conversationId
          ? await apiClient.sendConversationMessage(datasetId, conversationId, text)
          : await apiClient.createConversation(datasetId, text);
        if (!conversationId && response && response.conversation_id) {
          conversationId = String(response.conversation_id);
        }
        messages.push({
          id: response && response.turn_id ? String(response.turn_id) : `assistant-${++sequence}`,
          role: "assistant",
          intent: response && response.intent ? String(response.intent) : "UNKNOWN",
          status: response && response.status ? String(response.status) : "INVALID",
          text: response && response.message ? String(response.message) : "",
          visualization: response ? response.visualization || null : null,
          insight: response ? response.insight || null : null,
          clarificationOptions: response && Array.isArray(response.clarification_options)
            ? response.clarification_options.map(String) : [],
          createdAt: now(),
        });
        return response;
      } catch (requestError) {
        error = {
          code: requestError && requestError.code ? requestError.code : "request_failed",
          status: requestError && requestError.status ? requestError.status : null,
          message: requestError && requestError.message
            ? requestError.message : "Unable to process this request right now.",
        };
        return null;
      } finally {
        loading = false;
        emit();
      }
    }

    return { setDataset, reset, submit, getState: snapshot };
  }

  return { createConversationController };
});
