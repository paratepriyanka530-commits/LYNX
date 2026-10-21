/** Shared lifecycle controller for automatic and user-requested chart surfaces. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.LYNXVisualizationController = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function createVisualizationController({ adapter, legacyRenderer } = {}) {
    if (!adapter) throw new TypeError("adapter is required");

    function renderAll(scope, visualizations) {
      if (!scope) return [];
      const hosts = [...scope.querySelectorAll("[data-echarts-host]")];
      return hosts.map((host, index) => {
        const visualization = visualizations[index];
        if (!visualization) return { status: "EMPTY" };
        const result = adapter.render(host, visualization);
        if ((result.status === "ECHARTS_UNAVAILABLE" || !visualization.chart_spec) && legacyRenderer) {
          host.removeAttribute("data-echarts-host");
          legacyRenderer.render(host, visualization);
          host.dataset.rendererStatus = "LEGACY_FALLBACK";
          return { status: "LEGACY_FALLBACK" };
        }
        return result;
      });
    }

    function disposeWithin(scope) { adapter.disposeWithin(scope); }
    function resizeWithin(scope) { adapter.resizeWithin(scope); }
    return { renderAll, disposeWithin, resizeWithin };
  }

  return { createVisualizationController };
});
