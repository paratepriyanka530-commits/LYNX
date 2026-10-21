/** Deterministic display formatting for renderer-neutral chart hints. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.LYNXChartFormatter = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const numeric = (value) => value !== null && value !== "" && Number.isFinite(Number(value));

  function formatNumber(value, compact = false, maximumFractionDigits = 2) {
    if (!numeric(value)) return String(value == null ? "" : value);
    return new Intl.NumberFormat("en-US", {
      notation: compact ? "compact" : "standard",
      compactDisplay: "short",
      maximumFractionDigits,
    }).format(Number(value));
  }

  function formatValue(value, hint = "NUMBER", options = {}) {
    const kind = String(hint || "NUMBER").toUpperCase();
    if (kind === "DATE" || kind === "DATETIME") {
      const parsed = new Date(value);
      if (Number.isNaN(parsed.getTime())) return String(value == null ? "" : value);
      return new Intl.DateTimeFormat("en-US", kind === "DATE"
        ? { year: "numeric", month: "short", day: "numeric" }
        : { year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(parsed);
    }
    if (kind === "INTEGER") return formatNumber(value, Boolean(options.compact), 0);
    if (kind === "PERCENT") return numeric(value) ? `${formatNumber(value, false, 2)}%` : String(value ?? "");
    if (kind === "CURRENCY" && options.currency && /^[A-Z]{3}$/.test(options.currency)) {
      return new Intl.NumberFormat("en-US", {
        style: "currency", currency: options.currency,
        notation: options.compact ? "compact" : "standard", maximumFractionDigits: 2,
      }).format(Number(value));
    }
    if (kind === "DURATION") return numeric(value) ? `${formatNumber(value, false, 2)} ${options.unit || ""}`.trim() : String(value ?? "");
    return formatNumber(value, Boolean(options.compact), 2);
  }

  return { formatNumber, formatValue, numeric };
});
