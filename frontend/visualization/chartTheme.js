/** Professional light-background palettes; LYNX purple remains application chrome. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.LYNXChartTheme = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const PALETTES = Object.freeze({
    CATEGORICAL: ["#3E6B89", "#B2764B", "#657A55", "#8A6687", "#AA8A3D", "#4D8582", "#9A5E63", "#6372A0"],
    SEQUENTIAL: ["#DCE8ED", "#B9D2DC", "#8FB8C8", "#6399AF", "#3E768F", "#28576D"],
    DIVERGING: ["#A85751", "#D08A77", "#E7C8B8", "#E8E5DE", "#B7CFD1", "#72A4AA", "#39777E"],
    SINGLE_SERIES: ["#456F91"],
    MULTI_SERIES: ["#456F91", "#B4784F", "#667F5A", "#8B698C", "#4D8582", "#9C6267"],
    STATUS: ["#456F91", "#B8873B", "#8B5960", "#667F5A"],
  });

  function palette(role) { return PALETTES[String(role || "SINGLE_SERIES").toUpperCase()] || PALETTES.SINGLE_SERIES; }

  const BASE = Object.freeze({
    textStyle: { fontFamily: "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", color: "#37324C" },
    animationDuration: 350,
    animationDurationUpdate: 220,
    animationEasing: "cubicOut",
  });

  return { BASE, PALETTES, palette };
});
