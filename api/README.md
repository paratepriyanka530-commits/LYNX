# LYNX API — Visualization Endpoints (Pipeline 3 Integration)

The Visualization API provides high-performance, deterministic visualization generation and natural-language query resolution for tabular datasets.

> **Scope & Performance:**
> This API operates strictly on **already-cleaned data**. It does **NOT** run Pipeline 1 cleaning or invoke LLMs synchronously. Requests execute in milliseconds (<50ms).

---

## 1. Endpoints

### A. Automatic Mode
```http
GET /api/visualizations/{dataset_id}?max_visualizations=5
POST /api/visualizations/{dataset_id}
```
* **Description:** Loads the cleaned dataset from disk, executes Pipeline 3 Stages 1–6 (orchestrating intent-diverse candidate visualizations), enriches each item with Stage 9 factual XAI explanations, and returns the unified response contract.
* **Query Parameters:**
  * `max_visualizations` (optional integer, default `5`, min `1`, max `50`): Maximum number of visualizations to return.
* **Response (HTTP 200):**
```json
{
  "mode": "automatic",
  "status": "READY",
  "dataset_name": "pizza_sales",
  "visualization_count": 5,
  "visualizations": [
    {
      "rank": 1,
      "chart_type": "COUNT_BAR",
      "x_column": "pizza_category",
      "y_column": null,
      "score": 0.999965,
      "chart_data": {
        "chart_type": "COUNT_BAR",
        "x_column": "pizza_category",
        "y_column": null,
        "aggregation": "count",
        "points": [{"x": "Classic", "y": 14888}, ...],
        "bins": [],
        "signals": [...],
        "usable_row_count": 48620,
        "sampled": false,
        "has_data": true
      },
      "signals": [...],
      "explanation": {
        "why_this_chart": "Shows the distribution of observations across categories of 'pizza_category'.",
        "what_it_shows": "Displays 4 categories using observation counts.",
        "aggregation": "Semantic/default aggregation was applied.",
        "data_coverage": "4 displayed categories from 48,620 usable rows.",
        "evidence": ["'Classic' is highest with 14,888 observations.", ...]
      }
    }
  ],
  "signals": [...]
}
```

---

### B. Requested Mode
```http
POST /api/visualizations/{dataset_id}/query
Content-Type: application/json

{
  "request_text": "Median unit_price by pizza_category"
}
```
* **Description:** Resolves a natural-language visualization query across Stage 7 (parse/validate) $\rightarrow$ Stage 8 (resolve) $\rightarrow$ Stage 9 (explain).
* **Request Body:**
  * `request_text` (string, required, non-empty): The natural-language visualization query.
* **Response (HTTP 200):**
```json
{
  "mode": "requested",
  "request_text": "Median unit_price by pizza_category",
  "status": "READY",
  "dataset_name": "pizza_sales",
  "visualization_count": 1,
  "visualizations": [
    {
      "rank": 1,
      "chart_type": "BAR",
      "x_column": "pizza_category",
      "y_column": "unit_price",
      "score": null,
      "chart_data": {
        "chart_type": "BAR",
        "x_column": "pizza_category",
        "y_column": "unit_price",
        "aggregation": "median",
        "points": [{"x": "Classic", "y": 16.0}, ...],
        "bins": [],
        "signals": [...],
        "usable_row_count": 48620,
        "sampled": false,
        "has_data": true
      },
      "signals": [...],
      "explanation": {
        "why_this_chart": "Compares numeric values of 'unit_price' across categories of 'pizza_category'.",
        "what_it_shows": "Aggregates 'unit_price' by category using median.",
        "aggregation": "User-requested MEDIAN aggregation was applied.",
        "data_coverage": "4 displayed categories from 48,620 usable rows.",
        "evidence": [...]
      }
    }
  ],
  "signals": [...]
}
```

---

## 2. Status Code & Error Handling Rules

| Situation | HTTP Status | Response Format |
|---|---|---|
| Successful visualization | **200 OK** | Full response payload (`status: "READY"`) |
| No meaningful chart possible (e.g. constant/empty) | **200 OK** | Legitimate outcome (`status: "NO_MEANINGFUL_VISUALIZATION"`, `visualizations: []`) |
| Ambiguous query needing user clarification | **200 OK** | Legitimate outcome (`status: "NEEDS_CLARIFICATION"`, `clarification_options: [...]`) |
| Incompatible chart/aggregation request | **200 OK** | Legitimate outcome (`status: "INVALID"`, `errors: [...]`) |
| Unknown / unresolvable `dataset_id` | **404 Not Found** | `{"error": {"code": "dataset_not_found", "message": "..."}}` |
| Malformed body / missing `request_text` | **400 Bad Request** | `{"error": {"code": "malformed_request" | "missing_request_text", "message": "..."}}` |
| Internal read error | **500 Internal Error** | `{"error": {"code": "dataset_read_error", "message": "..."}}` |

---

## 3. Dataset Resolution Mapping

When a caller supplies a `dataset_id`, it is resolved to a cleaned CSV and optional cleaning report on disk via `api/dataset_resolver.py`:

1. **TabArena Benchmark Tasks:**
   * Formats: `"363612"`, `"task_363612"`, `"tabarena:363612"`
   * Resolves to: `benchmarks/tabarena/pipeline1_outputs/{task_id}/cleaned_dataset.csv`
   * Report: `benchmarks/tabarena/pipeline1_outputs/{task_id}/cleaning_report.json`
2. **Domain Cleaned Datasets:**
   * Formats: Exact filenames (`"pizza_sales"`, `"dataset_01_retail_transactions_raw"`) or aliases (`"pizza"`, `"retail"`, `"iot"`, `"survey"`, `"ecommerce"`)
   * Resolves to (primary): `storage/domain_datasets/{filename}.csv`
   * Resolves to (fallback): `tests/datasets/valid/{filename}.csv`
   * **First-time setup:** `storage/domain_datasets/` is excluded from git (regenerated from source CSVs).
     After a fresh checkout, run once before using domain-dataset endpoints:
     ```powershell
     Copy-Item tests/datasets/valid/*.csv storage/domain_datasets/
     ```
3. **Application Run Uploads:**
   * Formats: Valid UUID string (from `POST /api/runs`)
   * Resolves to: `{runtime_root}/{run_id}/pipeline1/cleaned_dataset.csv`  (default runtime root: `runtime/`)
   * Report: `{runtime_root}/{run_id}/pipeline1/cleaning_report.json`