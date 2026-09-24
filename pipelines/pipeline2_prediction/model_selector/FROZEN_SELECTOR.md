# LYNX Model Selector — Frozen Production Status

## Status: FROZEN

The model-selection research, representation validation, and final selector training are complete.
This selector is frozen and designated as immutable. Do not retrain, redesign, tune, or overwrite this selector or its supporting benchmark evidence.

---

## 1. Specification & Training Summary

| Attribute | Specification |
| :--- | :--- |
| **Selector Status** | **FROZEN (Production Immutable)** |
| **Final Selected Representation** | **Mode D** (Semantic dataset representation + shared task/target and candidate model context) |
| **Meta-Model Algorithm** | `RandomForestRegressor` |
| **Hyperparameters** | `n_estimators=100`, `max_depth=10`, `min_samples_leaf=2`, `random_state=42` |
| **Training Knowledge Source** | Authoritative $51 \times 10$ TabArena benchmark suite |
| **Source Benchmark Datasets** | 51 independent datasets |
| **Candidate Model Families** | 10 canonical tabular architectures |
| **Total Benchmark Observations** | 510 dataset–model pairs |
| **Successful Supervised Records** | **437 observations** |
| **Excluded Failed Observations** | **73 observations** (preserved for audit only; excluded from rank target to prevent zero-score distortion) |
| **Classification Supervision** | `balanced_accuracy` (higher is better) |
| **Regression Supervision** | `rmse` (lower is better) |
| **Learning Target Definition** | Within-dataset normalized relative rank in $[0.0, 1.0]$; empirical winner receives 1.0, worst receives 0.0 |

---

## 2. Validated Generalization Performance (Leave-One-Dataset-Out)

Across the 51 independent benchmark datasets evaluated under strict Leave-One-Dataset-Out (LODO) cross-validation:

* **Top-1 Accuracy:** **39.2%** (20 / 51 datasets) — *95% Bootstrap CI: [25.5%, 52.9%]*
* **Top-2 Accuracy:** **62.7%** (32 / 51 datasets) — *95% Bootstrap CI: [49.0%, 76.5%]*
* **Mean Normalized Regret:** **0.2593** — *95% Bootstrap CI: [0.1670, 0.3586]*
* **Statistical Superiority:** Semantic Mode $D$ achieves a $+13.7\%$ Top-2 accuracy advantage over structural baseline Mode $S$ ($p = 0.0358$).

---

## 3. Operational Invariants & Inference Contract

1. **Purpose:** The selector learns generalizable model suitability patterns conditioned strictly on dataset semantic, task, and candidate model capabilities.
2. **Zero Outcome Leakage:** Benchmark performance, metric values, runtimes, and failure statuses were used solely as training supervision. They are **never required or exposed** during inference on new datasets.
3. **External Production Verification:** Verified on the completely unseen external dataset `pizza_sales` (48,620 rows $\times$ 12 columns, regression):
   * Automated semantic profile generated via Ollama (`qwen2.5:7b`) without manual modification.
   * Full 10-model suitability ranking produced without retraining or benchmark execution.
   * Zero predictive models were trained during inference testing.

---

## 4. Immutable Artifact Registry & Cryptographic Hashes

The following files represent the complete, frozen research and production artifacts. Any future retraining, architecture modification, or feature change **must create a new selector version** rather than overwriting these artifacts or altering the historical benchmark evidence:

| Artifact Path | Description | SHA-256 Cryptographic Hash |
| :--- | :--- | :--- |
| [`lynx_model_selector.joblib`](file:///g:/My%20Drive/GitHub/LYNX/pipelines/pipeline2_prediction/model_selector/lynx_model_selector.joblib) | Production selector model bundle | `46ff89a509316389aa23b0f4b8083f2f83e8804759a4966e83e4616d59e06550` |
| [`final_selector_metadata.json`](file:///g:/My%20Drive/GitHub/LYNX/pipelines/pipeline2_prediction/model_selector/final_selector_metadata.json) | Comprehensive selector metadata & schema | `c096a62acefc7189b7d0e6ea2d443cf7b5b56d83e8a56cb873af07dece1abe03` |
| [`train_final_selector.py`](file:///g:/My%20Drive/GitHub/LYNX/pipelines/pipeline2_prediction/model_selector/train_final_selector.py) | Final production training script | Immutable reference implementation |
| [`selector_dataset.json`](file:///g:/My%20Drive/GitHub/LYNX/pipelines/pipeline2_prediction/model_selector/selector_dataset.json) | 510-record meta-learning training corpus | Immutable training input |
| [`selector_evaluation.json`](file:///g:/My%20Drive/GitHub/LYNX/pipelines/pipeline2_prediction/model_selector/selector_evaluation.json) | LODO evaluations across S, D, and S+D | Immutable experimental evidence |
| [`selector_statistics.json`](file:///g:/My%20Drive/GitHub/LYNX/pipelines/pipeline2_prediction/model_selector/selector_statistics.json) | 10,000 paired bootstrap resample results | Immutable statistical evidence |
| [`feasibility_semantic_evaluation.json`](file:///g:/My%20Drive/GitHub/LYNX/pipelines/pipeline2_prediction/model_selector/feasibility_semantic_evaluation.json) | Deterministic feasibility filter evaluation | Immutable hypothesis evidence |
| [`soft_feasibility_evaluation.json`](file:///g:/My%20Drive/GitHub/LYNX/pipelines/pipeline2_prediction/model_selector/soft_feasibility_evaluation.json) | Soft feasibility adjustment evaluation | Immutable sensitivity evidence |
| [`external_selector_test_01.json`](file:///g:/My%20Drive/GitHub/LYNX/pipelines/pipeline2_prediction/model_selector/external_selector_test_01.json) | Verification test on unseen dataset | Immutable verification audit |
| [`pipelines/pipeline2_prediction/training_data/model_performance.json`](file:///g:/My%20Drive/GitHub/LYNX/pipelines/pipeline2_prediction/training_data/model_performance.json) | Original 510-run benchmark ground truth | Immutable raw benchmark evidence |

---

## 5. Versioning Policy

> **MANDATE:**
> Any future retraining, hyperparameter refinement, or algorithmic alteration MUST be versioned independently (e.g., `v2_model_selector/` or `lynx_model_selector_v2.joblib`).
> Under no circumstances may this frozen artifact, its supporting LODO evaluations, or the underlying benchmark dataset be overwritten, renamed, normalized, or deleted.

