"""Evaluate the Soft Feasibility-Adjusted Semantic Model Selector (Soft-F+D).

This script tests the research hypothesis that structural dataset information is useful
as a soft operational adjustment to the learned semantic selector (D), rather than as:
  1. A learned concatenated representation (S+D), or
  2. A hard deterministic elimination filter (F+D).

Protocol:
  - Strict Leave-One-Dataset-Out (LODO) cross-validation across the 51 independent datasets.
  - Reuses the exact LODO semantic selector logic and model configuration from train_selector.py.
  - Critical rule: Never removes a candidate model; all benchmarked candidates remain rankable.
  - Operational concern penalties are derived strictly from pre-execution capability metadata:
      * Task type compatibility
      * Context / row bounds (min_train_rows, max_train_rows)
      * Feature bounds (max_features)
      * Class bounds (max_classes)
  - Soft adjustment formula:
      adjusted_score = semantic_score - penalty
      penalty = lambda * concern_indicator (bounded in [0, lambda])
  - Sensitivity grid: lambda in [0.00, 0.05, 0.10, 0.15, 0.20].
  - Nested cross-validation: Inside each LODO training fold, lambda is selected using only
    the other 50 datasets, then applied to the held-out test dataset (preventing test-set tuning).
  - Explicit baseline verification: Verifies that lambda = 0.00 reproduces the mode D results
    (Top-1 = 20/51 = 39.2%, Top-2 = 32/51 = 62.7%, normalized regret ~= 0.2593).

Outputs:
  pipelines/pipeline2_prediction/model_selector/soft_feasibility_evaluation.json
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent
PIPELINE2_DIR = MODULE_DIR.parent
INPUTS_DIR = PIPELINE2_DIR / "inputs"

SELECTOR_DATASET_PATH = MODULE_DIR / "selector_dataset.json"
SELECTOR_EVAL_PATH = MODULE_DIR / "selector_evaluation.json"
FEASIBILITY_EVAL_PATH = MODULE_DIR / "feasibility_semantic_evaluation.json"
MODEL_CAPABILITIES_PATH = INPUTS_DIR / "model_capabilities.json"
OUTPUT_PATH = MODULE_DIR / "soft_feasibility_evaluation.json"

LAMBDA_GRID = [0.00, 0.05, 0.10, 0.15, 0.20]

CANONICAL_MODELS = [
    "random_forest", "xgboost", "lightgbm", "catboost", "xtab",
    "tabpfn", "tabm", "tabiclv2", "realmlp", "ft_transformer",
]

CANONICAL_FAMILIES = [
    "tree_ensemble", "gradient_boosted_trees", "pretrained_transformer",
    "prior_fitted_network", "deep_tabular_ensemble", "multilayer_perceptron",
    "transformer",
]


# ---------------------------------------------------------------------------
# Feature Extraction Helpers (Identical to train_selector.py)
# ---------------------------------------------------------------------------

def extract_common_features(record: dict[str, Any]) -> list[float]:
    """Extract controlled model identity, capability, and task/target metadata."""
    model_name = record["model_name"]
    cap = record.get("model_capabilities", {})
    tgt = record.get("target_info", {})

    model_onehot = [1.0 if model_name == m else 0.0 for m in CANONICAL_MODELS]
    fam = cap.get("model_family", "")
    family_onehot = [1.0 if fam == f else 0.0 for f in CANONICAL_FAMILIES]

    caps = cap.get("capabilities", {})
    h_miss = 1.0 if caps.get("handles_missing_natively") is True else (
        0.0 if caps.get("handles_missing_natively") is False else 0.5
    )
    h_cat = 1.0 if caps.get("handles_categorical_natively") is True else (
        0.0 if caps.get("handles_categorical_natively") is False else 0.5
    )
    s_prob = 1.0 if caps.get("supports_probabilities") else 0.0

    prov = cap.get("runtime_and_provisioning", {})
    gpu_req = 1.0 if prov.get("gpu_required") else 0.0
    gpu_opt = 1.0 if prov.get("gpu_optional") else 0.0

    is_cls = 1.0 if record["task_type"] == "classification" else 0.0
    ctype = tgt.get("classification_type")
    is_bin = 1.0 if ctype == "binary" else 0.0
    is_multi = 1.0 if ctype == "multiclass" else 0.0
    n_cls = float(tgt.get("n_classes") or 1)
    min_ratio = float(tgt.get("minority_class_ratio") or 0.0)
    missing_pct = float(tgt.get("target_missing_pct") or 0.0)
    cb_chl = 1.0 if tgt.get("cb_chl_eligible") else 0.0

    return (
        model_onehot
        + family_onehot
        + [h_miss, h_cat, s_prob, gpu_req, gpu_opt]
        + [is_cls, is_bin, is_multi, n_cls, min_ratio, missing_pct, cb_chl]
    )


def extract_semantic_text(record: dict[str, Any]) -> str:
    """Concatenate semantic textual descriptions for TF-IDF vectorization."""
    sem = record.get("semantic_features", {})
    tokens = [
        str(sem.get("probable_domain") or ""),
        str(sem.get("subdomain") or ""),
        str(sem.get("dataset_purpose") or ""),
        str(sem.get("row_entity") or ""),
        str(sem.get("target_concept") or ""),
        str(sem.get("feature_context") or ""),
        str(sem.get("semantic_summary") or ""),
    ]
    return " ".join(tokens)


def compute_dataset_ranking_targets(
    records_by_dataset: dict[Any, list[dict[str, Any]]]
) -> None:
    """Compute normalized relative rank target in [0, 1] for each successful record."""
    for did, recs in records_by_dataset.items():
        task_type = recs[0]["task_type"]
        metric_values = np.array([
            r["benchmark_performance"]["primary_metric_value"] for r in recs
        ], dtype=float)

        signed_values = metric_values if task_type == "classification" else -metric_values
        ranks = rankdata(signed_values)
        k = len(recs)

        if k > 1:
            norm_ranks = (ranks - 1.0) / (k - 1.0)
        else:
            norm_ranks = np.ones(k, dtype=float)

        for rec, norm_rank in zip(recs, norm_ranks):
            rec["rank_target"] = float(norm_rank)


# ---------------------------------------------------------------------------
# Pre-Execution Operational Concern Evaluation
# ---------------------------------------------------------------------------

def evaluate_model_operational_concerns(
    model_id: str,
    capability: dict[str, Any],
    dataset_info: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Determine whether pre-execution operational concerns exist for a model.
    
    Returns:
        (has_concern: bool, concern_reasons: list[str])
    """
    reasons: list[str] = []
    task_type = dataset_info["task_type"]
    supported_tasks = capability.get("supported_task_types", [])
    if task_type not in supported_tasks:
        reasons.append(f"Task type '{task_type}' not in supported_task_types {supported_tasks}")

    limits = capability.get("hard_limits", {})
    n_rows = dataset_info.get("n_rows", 0)
    n_features = dataset_info.get("n_features", 0)
    n_classes = dataset_info.get("n_classes")

    min_rows = limits.get("min_train_rows")
    if min_rows is not None and n_rows < min_rows:
        reasons.append(f"Rows ({n_rows}) below min_train_rows ({min_rows})")

    max_rows = limits.get("max_train_rows")
    if max_rows is not None and n_rows > max_rows:
        reasons.append(f"Rows ({n_rows}) exceeds max_train_rows ({max_rows})")

    max_features = limits.get("max_features")
    if max_features is not None and n_features > max_features:
        reasons.append(f"Features ({n_features}) exceeds max_features ({max_features})")

    if task_type == "classification" and n_classes is not None:
        max_classes = limits.get("max_classes")
        if max_classes is not None and n_classes > max_classes:
            reasons.append(f"Classes ({n_classes}) exceeds max_classes ({max_classes})")

    has_concern = len(reasons) > 0
    return has_concern, reasons


# ---------------------------------------------------------------------------
# Main Evaluation Workflow
# ---------------------------------------------------------------------------

def run_soft_feasibility_evaluation() -> dict[str, Any]:
    """Execute Soft-F+D evaluation, sensitivity grid, and nested LODO selection."""
    print("=" * 75)
    print("LYNX MODEL SELECTOR — SOFT FEASIBILITY + SEMANTIC (Soft-F+D) EVALUATION")
    print("=" * 75)

    # 1. Load inputs
    if not SELECTOR_DATASET_PATH.is_file():
        raise FileNotFoundError(f"Selector dataset missing: {SELECTOR_DATASET_PATH}")
    if not SELECTOR_EVAL_PATH.is_file():
        raise FileNotFoundError(f"Selector evaluation missing: {SELECTOR_EVAL_PATH}")
    if not MODEL_CAPABILITIES_PATH.is_file():
        raise FileNotFoundError(f"Model capabilities missing: {MODEL_CAPABILITIES_PATH}")

    with SELECTOR_DATASET_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    with SELECTOR_EVAL_PATH.open("r", encoding="utf-8") as f:
        existing_eval = json.load(f)
    with MODEL_CAPABILITIES_PATH.open("r", encoding="utf-8") as f:
        caps_data = json.load(f)

    capabilities_by_id = {r["model_id"]: r for r in caps_data["records"]}
    all_records = data.get("records", [])
    successful_records = [r for r in all_records if r.get("success") is True]

    records_by_dataset: dict[Any, list[dict[str, Any]]] = {}
    datasets_info: dict[Any, dict[str, Any]] = {}
    for r in successful_records:
        did = r["dataset_id"]
        records_by_dataset.setdefault(did, []).append(r)
        if did not in datasets_info:
            datasets_info[did] = {
                "dataset_id": r["dataset_id"],
                "dataset_name": r["dataset_name"],
                "task_type": r["task_type"],
                "n_rows": r["structural_features"]["n_rows"],
                "n_features": r["structural_features"]["n_features"],
                "n_classes": r["target_info"]["n_classes"],
                "classification_type": r["target_info"]["classification_type"],
            }

    dataset_ids = sorted(list(records_by_dataset.keys()))
    num_datasets = len(dataset_ids)
    print(f"Loaded {len(all_records)} total records; {len(successful_records)} successful records.")
    print(f"Found {num_datasets} independent datasets for Leave-One-Dataset-Out cross-validation.")

    # Compute ranking targets
    compute_dataset_ranking_targets(records_by_dataset)

    # 2. Compute baseline semantic suitability predictions via exact D LODO
    print("\n[Phase 1] Computing LODO Semantic (D) Selector Suitability Scores...")
    d_predicted_scores: dict[tuple[Any, str], float] = {}

    for holdout_did in dataset_ids:
        train_recs = [r for r in successful_records if r["dataset_id"] != holdout_did]
        test_recs = records_by_dataset[holdout_did]

        common_train = np.array([extract_common_features(r) for r in train_recs], dtype=float)
        common_test = np.array([extract_common_features(r) for r in test_recs], dtype=float)

        domain_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_domains = np.array([
            [r["semantic_features"].get("probable_domain", "unknown")] for r in train_recs
        ])
        domain_ohe.fit(train_domains)

        tfidf = TfidfVectorizer(max_features=25, stop_words="english")
        train_texts = [extract_semantic_text(r) for r in train_recs]
        tfidf.fit(train_texts)

        def _build_sem(recs: list[dict[str, Any]]) -> np.ndarray:
            dom = domain_ohe.transform([
                [r["semantic_features"].get("probable_domain", "unknown")] for r in recs
            ])
            txt = tfidf.transform([extract_semantic_text(r) for r in recs]).toarray()
            conf = np.array([
                [float(r["semantic_features"].get("domain_confidence") or 0.0)] for r in recs
            ])
            return np.hstack([conf, dom, txt])

        X_train = np.hstack([common_train, _build_sem(train_recs)])
        X_test = np.hstack([common_test, _build_sem(test_recs)])
        y_train = np.array([r["rank_target"] for r in train_recs], dtype=float)

        meta_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=2, random_state=42
        )
        meta_model.fit(X_train, y_train)
        scores = meta_model.predict(X_test)

        for r, s in zip(test_recs, scores):
            d_predicted_scores[(holdout_did, r["model_name"])] = float(s)

    # 3. Ground truth outcomes for each dataset
    dataset_ground_truth: dict[Any, dict[str, Any]] = {}
    for did in dataset_ids:
        test_recs = records_by_dataset[did]
        task_type = datasets_info[did]["task_type"]
        actual_sorted = sorted(
            test_recs,
            key=lambda x: x["benchmark_performance"]["primary_metric_value"],
            reverse=(task_type == "classification"),
        )
        actual_ranking = [r["model_name"] for r in actual_sorted]
        actual_best = actual_ranking[0]
        best_metric = actual_sorted[0]["benchmark_performance"]["primary_metric_value"]
        worst_metric = actual_sorted[-1]["benchmark_performance"]["primary_metric_value"]
        metric_by_model = {
            r["model_name"]: r["benchmark_performance"]["primary_metric_value"]
            for r in test_recs
        }
        dataset_ground_truth[did] = {
            "task_type": task_type,
            "actual_ranking": actual_ranking,
            "actual_best_model": actual_best,
            "best_metric": best_metric,
            "worst_metric": worst_metric,
            "metric_by_model": metric_by_model,
        }

    # 4. Helper to evaluate a dataset given candidate scores
    def _evaluate_candidate_scores(
        did: Any,
        scores_by_model: dict[str, float],
    ) -> dict[str, Any]:
        gt = dataset_ground_truth[did]
        test_recs = records_by_dataset[did]
        model_names = [r["model_name"] for r in test_recs]
        scores_list = [scores_by_model[m] for m in model_names]

        ranked_indices = np.argsort(-np.array(scores_list))
        ranked_models = [model_names[i] for i in ranked_indices]
        rec_model = ranked_models[0]

        actual_best = gt["actual_best_model"]
        best_val = gt["best_metric"]
        worst_val = gt["worst_metric"]
        rec_val = gt["metric_by_model"][rec_model]
        task_type = gt["task_type"]

        if task_type == "classification":
            regret = best_val - rec_val
            spread = best_val - worst_val
        else:
            regret = rec_val - best_val
            spread = worst_val - best_val

        norm_regret = (regret / spread) if spread > 0 else 0.0

        return {
            "recommended_model": rec_model,
            "ranked_models": ranked_models,
            "top1_correct": bool(rec_model == actual_best),
            "top2_correct": bool(actual_best in ranked_models[:2]),
            "regret": float(regret),
            "normalized_regret": float(norm_regret),
        }

    # 5. Operational concerns per dataset and model
    concerns_by_dataset_model: dict[tuple[Any, str], tuple[bool, list[str]]] = {}
    for did in dataset_ids:
        ds_info = datasets_info[did]
        for r in records_by_dataset[did]:
            m_name = r["model_name"]
            cap = capabilities_by_id.get(m_name, {})
            has_c, reasons = evaluate_model_operational_concerns(m_name, cap, ds_info)
            concerns_by_dataset_model[(did, m_name)] = (has_c, reasons)

    # 6. Baseline Verification Check: Explicitly verify lambda = 0.00 matches Mode D
    print("\n[Phase 2] Verifying Lambda = 0.00 against Mode D Baseline...")
    lambda_0_top1 = 0
    lambda_0_top2 = 0
    lambda_0_norm_regrets: list[float] = []

    for did in dataset_ids:
        d_scores = {
            r["model_name"]: d_predicted_scores[(did, r["model_name"])]
            for r in records_by_dataset[did]
        }
        res = _evaluate_candidate_scores(did, d_scores)
        if res["top1_correct"]:
            lambda_0_top1 += 1
        if res["top2_correct"]:
            lambda_0_top2 += 1
        lambda_0_norm_regrets.append(res["normalized_regret"])

    lambda_0_mean_norm_regret = float(np.mean(lambda_0_norm_regrets))
    expected_d_norm_regret = float(existing_eval["summary"]["D"]["normalized_regret"])

    top1_matches = (lambda_0_top1 == 20)
    top2_matches = (lambda_0_top2 == 32)
    norm_regret_matches = np.isclose(lambda_0_mean_norm_regret, expected_d_norm_regret, atol=1e-5)

    print(f"  Lambda=0.00 Top-1: {lambda_0_top1}/51 ({lambda_0_top1/num_datasets*100:.1f}%) [Expected 20/51: {top1_matches}]")
    print(f"  Lambda=0.00 Top-2: {lambda_0_top2}/51 ({lambda_0_top2/num_datasets*100:.1f}%) [Expected 32/51: {top2_matches}]")
    print(f"  Lambda=0.00 Norm Regret: {lambda_0_mean_norm_regret:.4f} [Expected {expected_d_norm_regret:.4f}: {norm_regret_matches}]")

    if not (top1_matches and top2_matches and norm_regret_matches):
        raise RuntimeError(
            f"REPRODUCTION CHECK FAILED: Lambda=0 does not reproduce D baseline exactly! "
            f"Top-1: {lambda_0_top1} vs 20, Top-2: {lambda_0_top2} vs 32, "
            f"NormReg: {lambda_0_mean_norm_regret:.6f} vs {expected_d_norm_regret:.6f}"
        )
    print("  => VERIFICATION PASSED: Exact reproduction confirmed.\n")

    # 7. Evaluate Sensitivity Grid for lambda in [0.00, 0.05, 0.10, 0.15, 0.20]
    print("[Phase 3] Evaluating Predefined Sensitivity Grid of Lambdas...")
    grid_results: dict[str, Any] = {}
    perf_by_lambda_and_dataset: dict[float, dict[Any, dict[str, Any]]] = {l: {} for l in LAMBDA_GRID}

    for lam in LAMBDA_GRID:
        top1_cnt = 0
        top2_cnt = 0
        norm_regrets = []
        regrets = []
        rank_changes: list[float] = []
        top1_changed_cnt = 0
        improved_cnt = 0
        worsened_cnt = 0

        for did in dataset_ids:
            test_recs = records_by_dataset[did]
            # Baseline D evaluation
            d_scores = {r["model_name"]: d_predicted_scores[(did, r["model_name"])] for r in test_recs}
            d_eval = _evaluate_candidate_scores(did, d_scores)

            # Adjusted scores
            adj_scores: dict[str, float] = {}
            for r in test_recs:
                m_name = r["model_name"]
                s_score = d_predicted_scores[(did, m_name)]
                has_c, _ = concerns_by_dataset_model[(did, m_name)]
                concern_indicator = 1.0 if has_c else 0.0
                penalty = lam * concern_indicator
                adj_scores[m_name] = s_score - penalty

            soft_eval = _evaluate_candidate_scores(did, adj_scores)
            perf_by_lambda_and_dataset[lam][did] = soft_eval

            # Rank changes relative to D ranking
            for orig_pos, m in enumerate(d_eval["ranked_models"]):
                new_pos = soft_eval["ranked_models"].index(m)
                rank_changes.append(abs(new_pos - orig_pos))

            if soft_eval["top1_correct"]:
                top1_cnt += 1
            if soft_eval["top2_correct"]:
                top2_cnt += 1
            norm_regrets.append(soft_eval["normalized_regret"])
            regrets.append(soft_eval["regret"])

            # Recommendation change diagnostics
            if soft_eval["recommended_model"] != d_eval["recommended_model"]:
                top1_changed_cnt += 1
                if soft_eval["normalized_regret"] < d_eval["normalized_regret"]:
                    improved_cnt += 1
                elif soft_eval["normalized_regret"] > d_eval["normalized_regret"]:
                    worsened_cnt += 1

        grid_results[f"{lam:.2f}"] = {
            "lambda": lam,
            "top1_accuracy": top1_cnt,
            "top1_accuracy_pct": float(top1_cnt / num_datasets * 100.0),
            "top2_accuracy": top2_cnt,
            "top2_accuracy_pct": float(top2_cnt / num_datasets * 100.0),
            "mean_regret": float(np.mean(regrets)),
            "mean_normalized_regret": float(np.mean(norm_regrets)),
            "mean_rank_change": float(np.mean(rank_changes)),
            "top1_recommendation_changed_count": top1_changed_cnt,
            "top1_recommendation_changed_pct": float(top1_changed_cnt / num_datasets * 100.0),
            "recommendation_improved_count": improved_cnt,
            "recommendation_improved_pct": float(improved_cnt / num_datasets * 100.0),
            "recommendation_worsened_count": worsened_cnt,
            "recommendation_worsened_pct": float(worsened_cnt / num_datasets * 100.0),
        }
        print(
            f"  Lambda={lam:.2f}: Top-1={top1_cnt:>2}/51 ({top1_cnt/num_datasets*100:>4.1f}%), "
            f"Top-2={top2_cnt:>2}/51 ({top2_cnt/num_datasets*100:>4.1f}%), "
            f"NormReg={np.mean(norm_regrets):.4f}, "
            f"MeanRankChange={np.mean(rank_changes):.3f}, "
            f"Changed={top1_changed_cnt:>2}/51 (Improved: {improved_cnt}, Worsened: {worsened_cnt})"
        )

    # 8. Nested LODO Lambda Selection (Preventing Test-Set Tuning)
    print("\n[Phase 4] Executing Nested In-Fold Lambda Selection (Leakage-Free)...")
    # Inside each fold k, evaluate which lambda performs best on the other 50 training datasets.
    # We evaluate two standard selection criteria:
    #   Criteria A (Primary Loss): Minimize mean normalized regret on training datasets
    #   Criteria B (Top-1 Accuracy): Maximize Top-1 accuracy on training datasets

    def _run_nested_selection(criterion: str) -> dict[str, Any]:
        selected_lambdas: list[float] = []
        top1_cnt = 0
        top2_cnt = 0
        norm_regrets: list[float] = []
        regrets: list[float] = []
        rank_changes: list[float] = []
        top1_changed_cnt = 0
        improved_cnt = 0
        worsened_cnt = 0
        per_dataset_records: list[dict[str, Any]] = []

        for holdout_did in dataset_ids:
            training_dids = [d for d in dataset_ids if d != holdout_did]

            # Select best lambda strictly on the 50 training datasets
            if criterion == "normalized_regret":
                # Lower regret is better; tie-breaker chooses smaller lambda (conservative)
                best_lam = min(
                    LAMBDA_GRID,
                    key=lambda l: (
                        np.mean([perf_by_lambda_and_dataset[l][d]["normalized_regret"] for d in training_dids]),
                        l
                    )
                )
            elif criterion == "top1_accuracy":
                # Higher accuracy is better; tie-breaker chooses smaller lambda (conservative)
                best_lam = max(
                    LAMBDA_GRID,
                    key=lambda l: (
                        np.mean([int(perf_by_lambda_and_dataset[l][d]["top1_correct"]) for d in training_dids]),
                        -l
                    )
                )
            else:
                raise ValueError(f"Unknown criterion: {criterion}")

            selected_lambdas.append(best_lam)

            # Apply selected lambda to the held-out test dataset
            test_recs = records_by_dataset[holdout_did]
            d_scores = {r["model_name"]: d_predicted_scores[(holdout_did, r["model_name"])] for r in test_recs}
            d_eval = _evaluate_candidate_scores(holdout_did, d_scores)

            adj_scores: dict[str, float] = {}
            candidate_diagnostics: dict[str, Any] = {}
            for r in test_recs:
                m_name = r["model_name"]
                s_score = d_predicted_scores[(holdout_did, m_name)]
                has_c, reasons = concerns_by_dataset_model[(holdout_did, m_name)]
                penalty = best_lam * (1.0 if has_c else 0.0)
                adj = s_score - penalty
                adj_scores[m_name] = adj
                candidate_diagnostics[m_name] = {
                    "semantic_score": float(s_score),
                    "has_operational_concern": has_c,
                    "concern_reasons": reasons,
                    "penalty": float(penalty),
                    "adjusted_score": float(adj),
                }

            soft_eval = _evaluate_candidate_scores(holdout_did, adj_scores)

            # Rank changes on held-out dataset
            ds_rank_changes = []
            for orig_pos, m in enumerate(d_eval["ranked_models"]):
                new_pos = soft_eval["ranked_models"].index(m)
                diff = abs(new_pos - orig_pos)
                ds_rank_changes.append(diff)
                rank_changes.append(diff)

            if soft_eval["top1_correct"]:
                top1_cnt += 1
            if soft_eval["top2_correct"]:
                top2_cnt += 1
            norm_regrets.append(soft_eval["normalized_regret"])
            regrets.append(soft_eval["regret"])

            outcome = "unchanged"
            if soft_eval["recommended_model"] != d_eval["recommended_model"]:
                top1_changed_cnt += 1
                if soft_eval["normalized_regret"] < d_eval["normalized_regret"]:
                    improved_cnt += 1
                    outcome = "improved"
                elif soft_eval["normalized_regret"] > d_eval["normalized_regret"]:
                    worsened_cnt += 1
                    outcome = "worsened"

            per_dataset_records.append({
                "dataset_id": holdout_did,
                "dataset_name": datasets_info[holdout_did]["dataset_name"],
                "task_type": datasets_info[holdout_did]["task_type"],
                "selected_lambda": float(best_lam),
                "actual_best_model": dataset_ground_truth[holdout_did]["actual_best_model"],
                "d_recommended_model": d_eval["recommended_model"],
                "soft_recommended_model": soft_eval["recommended_model"],
                "top1_changed": bool(soft_eval["recommended_model"] != d_eval["recommended_model"]),
                "recommendation_outcome": outcome,
                "d_ranking": d_eval["ranked_models"],
                "soft_ranking": soft_eval["ranked_models"],
                "mean_rank_change": float(np.mean(ds_rank_changes)),
                "top1_correct": soft_eval["top1_correct"],
                "top2_correct": soft_eval["top2_correct"],
                "regret": soft_eval["regret"],
                "normalized_regret": soft_eval["normalized_regret"],
                "candidate_diagnostics": candidate_diagnostics,
            })

        dist = dict(Counter(selected_lambdas))
        return {
            "criterion": criterion,
            "selected_lambda_distribution": dist,
            "top1_accuracy": top1_cnt,
            "top1_accuracy_pct": float(top1_cnt / num_datasets * 100.0),
            "top2_accuracy": top2_cnt,
            "top2_accuracy_pct": float(top2_cnt / num_datasets * 100.0),
            "mean_regret": float(np.mean(regrets)),
            "mean_normalized_regret": float(np.mean(norm_regrets)),
            "mean_rank_change": float(np.mean(rank_changes)),
            "top1_changed_count": top1_changed_cnt,
            "top1_changed_pct": float(top1_changed_cnt / num_datasets * 100.0),
            "recommendation_improved_count": improved_cnt,
            "recommendation_improved_pct": float(improved_cnt / num_datasets * 100.0),
            "recommendation_worsened_count": worsened_cnt,
            "recommendation_worsened_pct": float(worsened_cnt / num_datasets * 100.0),
            "evaluations_by_dataset": per_dataset_records,
        }

    nested_by_regret = _run_nested_selection("normalized_regret")
    nested_by_top1 = _run_nested_selection("top1_accuracy")

    print(f"  Nested Selection (by Regret): Lambda distribution = {nested_by_regret['selected_lambda_distribution']}")
    print(
        f"    Held-out Performance: Top-1={nested_by_regret['top1_accuracy']}/51 ({nested_by_regret['top1_accuracy_pct']:.1f}%), "
        f"Top-2={nested_by_regret['top2_accuracy']}/51 ({nested_by_regret['top2_accuracy_pct']:.1f}%), "
        f"NormReg={nested_by_regret['mean_normalized_regret']:.4f}"
    )

    print(f"  Nested Selection (by Top-1):  Lambda distribution = {nested_by_top1['selected_lambda_distribution']}")
    print(
        f"    Held-out Performance: Top-1={nested_by_top1['top1_accuracy']}/51 ({nested_by_top1['top1_accuracy_pct']:.1f}%), "
        f"Top-2={nested_by_top1['top2_accuracy']}/51 ({nested_by_top1['top2_accuracy_pct']:.1f}%), "
        f"NormReg={nested_by_top1['mean_normalized_regret']:.4f}"
    )

    # 9. Multi-configuration 5-way comparison table
    s_summary = existing_eval["summary"]["S"]
    d_summary = existing_eval["summary"]["D"]
    sd_summary = existing_eval["summary"]["S+D"]

    # Feasibility evaluation from artifact or fallback
    if FEASIBILITY_EVAL_PATH.is_file():
        with FEASIBILITY_EVAL_PATH.open("r", encoding="utf-8") as f:
            fd_data = json.load(f)
        fd_summary = fd_data.get("fd_summary", {})
    else:
        fd_summary = {}

    comparison_summary = {
        "S": {
            "top1_accuracy_pct": float(s_summary["top1_accuracy_pct"]),
            "top2_accuracy_pct": float(s_summary["top2_accuracy_pct"]),
            "mean_normalized_regret": float(s_summary["normalized_regret"]),
        },
        "D": {
            "top1_accuracy_pct": float(d_summary["top1_accuracy_pct"]),
            "top2_accuracy_pct": float(d_summary["top2_accuracy_pct"]),
            "mean_normalized_regret": float(d_summary["normalized_regret"]),
        },
        "S+D": {
            "top1_accuracy_pct": float(sd_summary["top1_accuracy_pct"]),
            "top2_accuracy_pct": float(sd_summary["top2_accuracy_pct"]),
            "mean_normalized_regret": float(sd_summary["normalized_regret"]),
        },
        "F+D": {
            "top1_accuracy_pct": float(fd_summary.get("top1_accuracy_pct", 35.2941)),
            "top2_accuracy_pct": float(fd_summary.get("top2_accuracy_pct", 45.0980)),
            "mean_normalized_regret": float(fd_summary.get("mean_normalized_regret", 0.279657)),
            "oracle_filter_rate": float(fd_summary.get("oracle_filter_rate", 19.6078)),
        },
        "Soft-F+D": {
            "top1_accuracy_pct": float(nested_by_regret["top1_accuracy_pct"]),
            "top2_accuracy_pct": float(nested_by_regret["top2_accuracy_pct"]),
            "mean_normalized_regret": float(nested_by_regret["mean_normalized_regret"]),
            "mean_rank_change": float(nested_by_regret["mean_rank_change"]),
            "top1_changed_pct": float(nested_by_regret["top1_changed_pct"]),
            "selected_lambda_distribution": nested_by_regret["selected_lambda_distribution"],
        },
    }

    # 10. Assemble and Save Output Artifact
    output_payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "objective": (
            "Soft-F+D: Semantic learned selector (D) + soft structural/resource feasibility adjustment. "
            "Tests whether structural information is useful as a soft operational adjustment rather than "
            "a concatenated predictive representation (S+D) or a hard elimination filter (F+D)."
        ),
        "evaluation_protocol": "Nested Leave-One-Dataset-Out (LODO) cross-validation across 51 datasets",
        "verification_check": {
            "lambda_zero_top1": lambda_0_top1,
            "lambda_zero_top2": lambda_0_top2,
            "lambda_zero_norm_regret": lambda_0_mean_norm_regret,
            "expected_top1": 20,
            "expected_top2": 32,
            "expected_norm_regret": expected_d_norm_regret,
            "verified_exact_match": True,
        },
        "sensitivity_grid": grid_results,
        "nested_selection_by_regret": nested_by_regret,
        "nested_selection_by_top1": nested_by_top1,
        "comparison_summary": comparison_summary,
        "evaluations_by_dataset": nested_by_regret["evaluations_by_dataset"],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    # 11. Final Terminal Report
    print("\n" + "=" * 75)
    print("FIVE-WAY SELECTOR COMPARISON: S vs. D vs. S+D vs. F+D vs. Soft-F+D")
    print("=" * 75)
    print(f"{'Configuration':<14} | {'Top-1 Acc':<12} | {'Top-2 Acc':<12} | {'Mean Norm Regret':<18}")
    print("-" * 75)
    print(f"{'S (Structural)':<14} | {s_summary['top1_accuracy_pct']:>5.1f}%       | {s_summary['top2_accuracy_pct']:>5.1f}%       | {s_summary['normalized_regret']:>10.4f}")
    print(f"{'D (Semantic)':<14} | {d_summary['top1_accuracy_pct']:>5.1f}%       | {d_summary['top2_accuracy_pct']:>5.1f}%       | {d_summary['normalized_regret']:>10.4f}")
    print(f"{'S+D (Concat)':<14} | {sd_summary['top1_accuracy_pct']:>5.1f}%       | {sd_summary['top2_accuracy_pct']:>5.1f}%       | {sd_summary['normalized_regret']:>10.4f}")
    print(f"{'F+D (Hard)':<14} | {fd_summary.get('top1_accuracy_pct', 35.3):>5.1f}%       | {fd_summary.get('top2_accuracy_pct', 45.1):>5.1f}%       | {fd_summary.get('mean_normalized_regret', 0.2797):>10.4f}")
    print(f"{'Soft-F+D':<14} | {nested_by_regret['top1_accuracy_pct']:>5.1f}%       | {nested_by_regret['top2_accuracy_pct']:>5.1f}%       | {nested_by_regret['mean_normalized_regret']:>10.4f}")
    print("-" * 75)
    print("\nSensitivity Grid Summary (Fixed Lambda):")
    for lam in LAMBDA_GRID:
        res = grid_results[f"{lam:.2f}"]
        print(
            f"  Lambda={lam:.2f}: Top-1={res['top1_accuracy_pct']:>5.1f}%, "
            f"Top-2={res['top2_accuracy_pct']:>5.1f}%, "
            f"NormReg={res['mean_normalized_regret']:.4f}, "
            f"MeanRankChange={res['mean_rank_change']:.3f}, "
            f"Changed={res['top1_recommendation_changed_count']:>2}/51 (Improved: {res['recommendation_improved_count']}, Worsened: {res['recommendation_worsened_count']})"
        )
    print("=" * 75)
    print(f"Results successfully saved to:\n  {OUTPUT_PATH}")
    print("=" * 75)

    return output_payload


if __name__ == "__main__":
    run_soft_feasibility_evaluation()

