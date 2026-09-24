"""Train and evaluate the LYNX meta-level model selector via Leave-One-Dataset-Out (LODO).

This module implements the primary learned model selector for LYNX.
It evaluates three feature representations:
  - S:   Structural/statistical dataset features
  - D:   Semantic dataset features
  - S+D: Combined structural and semantic features

Task/target information and model identity are held constant across all three modes
to isolate the impact of dataset representation on recommendation quality.

Protocol:
  - Strict Leave-One-Dataset-Out (LODO) over the 51 independent datasets.
  - Leakage prevention: encoders (OneHotEncoder, TfidfVectorizer) are fitted strictly
    inside each training fold; test fold features are transformed without refit.
  - Targets are normalized relative ranks derived from the benchmark primary metric:
      * Classification: higher balanced_accuracy is better
      * Regression: lower RMSE is better
  - Failures are excluded from performance training and preserved for audit only.
"""

from __future__ import annotations

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
# Paths and Canonical Constants
# ---------------------------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent
DATASET_PATH = MODULE_DIR / "selector_dataset.json"
OUTPUT_PATH = MODULE_DIR / "selector_evaluation.json"

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
# Feature Extraction Helpers
# ---------------------------------------------------------------------------

def extract_common_features(record: dict[str, Any]) -> list[float]:
    """Extract controlled model identity, capability, and task/target metadata.
    
    These features are shared identically across S, D, and S+D representations.
    """
    model_name = record["model_name"]
    cap = record.get("model_capabilities", {})
    tgt = record.get("target_info", {})

    # 1. Model identity one-hot (10 features)
    model_onehot = [1.0 if model_name == m else 0.0 for m in CANONICAL_MODELS]

    # 2. Model family one-hot (7 features)
    fam = cap.get("model_family", "")
    family_onehot = [1.0 if fam == f else 0.0 for f in CANONICAL_FAMILIES]

    # 3. Model capabilities (5 features)
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

    # 4. Task and target metadata (7 features)
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


def extract_structural_features(record: dict[str, Any]) -> list[float]:
    """Extract structural/statistical dataset features (S vector)."""
    s = record.get("structural_features", {})
    return [
        np.log1p(float(s.get("n_rows") or 0)),
        np.log1p(float(s.get("n_features") or 0)),
        float(s.get("numeric_feature_ratio") or 0.0),
        float(s.get("categorical_feature_ratio") or 0.0),
        float(s.get("boolean_feature_ratio") or 0.0),
        float(s.get("datetime_feature_ratio") or 0.0),
        float(s.get("other_feature_ratio") or 0.0),
        float(s.get("overall_missing_pct") or 0.0),
        float(s.get("features_with_missing_ratio") or 0.0),
        float(s.get("rows_with_missing_ratio") or 0.0),
        np.log1p(float(s.get("mean_feature_cardinality") or 0.0)),
        np.log1p(float(s.get("median_feature_cardinality") or 0.0)),
        float(s.get("mean_unique_ratio") or 0.0),
        float(s.get("high_cardinality_feature_ratio") or 0.0),
    ]


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


# ---------------------------------------------------------------------------
# Ranking Target Construction
# ---------------------------------------------------------------------------

def compute_dataset_ranking_targets(
    records_by_dataset: dict[Any, list[dict[str, Any]]]
) -> None:
    """Compute normalized relative rank target in [0, 1] for each successful record.
    
    Metric Direction Rules (matching authoritative benchmark specifications):
      - Classification: primary_metric_name == 'balanced_accuracy' -> Higher is better
      - Regression:     primary_metric_name == 'rmse'              -> Lower is better
      
    For each dataset with K successful models:
      - The actual best model receives a target of 1.0
      - The actual worst model receives a target of 0.0
      - Intermediate models receive linearly spaced rank values in [0, 1]
    """
    for did, recs in records_by_dataset.items():
        task_type = recs[0]["task_type"]
        metric_values = np.array([
            r["benchmark_performance"]["primary_metric_value"] for r in recs
        ], dtype=float)

        # Invert sign for regression so higher signed value always indicates superior performance
        signed_values = metric_values if task_type == "classification" else -metric_values
        ranks = rankdata(signed_values)  # 1-based, higher value gets larger rank
        k = len(recs)

        if k > 1:
            norm_ranks = (ranks - 1.0) / (k - 1.0)
        else:
            norm_ranks = np.ones(k, dtype=float)

        for rec, norm_rank in zip(recs, norm_ranks):
            rec["rank_target"] = float(norm_rank)


# ---------------------------------------------------------------------------
# Leave-One-Dataset-Out (LODO) Evaluation Procedure
# ---------------------------------------------------------------------------

def run_lodo_evaluation() -> dict[str, Any]:
    """Execute Leave-One-Dataset-Out cross-validation across S, D, and S+D modes."""
    print("=" * 70)
    print("LYNX MODEL SELECTOR — LEAVE-ONE-DATASET-OUT (LODO) BENCHMARK")
    print("=" * 70)

    # 1. Load dataset
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Selector dataset not found: {DATASET_PATH}")
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    all_records = data.get("records", [])
    successful_records = [r for r in all_records if r.get("success") is True]

    print(f"Loaded {len(all_records)} total records.")
    print(f"Retained {len(successful_records)} successful records for training.")

    # 2. Group records by dataset ID
    records_by_dataset: dict[Any, list[dict[str, Any]]] = {}
    for r in successful_records:
        did = r["dataset_id"]
        records_by_dataset.setdefault(did, []).append(r)

    dataset_ids = sorted(list(records_by_dataset.keys()))
    num_datasets = len(dataset_ids)
    print(f"Identified {num_datasets} independent datasets for LODO evaluation.")

    # 3. Compute normalized rank targets
    compute_dataset_ranking_targets(records_by_dataset)

    # 4. Evaluation Loop across S, D, S+D representations
    modes = ["S", "D", "S+D"]
    summary_results: dict[str, Any] = {}
    detailed_evaluations: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}

    for mode in modes:
        print(f"\n--- Running LODO Validation: Mode {mode} ---")
        top1_correct_count = 0
        top2_correct_count = 0
        all_regrets: list[float] = []
        all_norm_regrets: list[float] = []

        for fold_idx, holdout_did in enumerate(dataset_ids, start=1):
            train_recs = [r for r in successful_records if r["dataset_id"] != holdout_did]
            test_recs = records_by_dataset[holdout_did]

            # Common baseline features (Model ID + Capabilities + Task Metadata)
            common_train = np.array([extract_common_features(r) for r in train_recs], dtype=float)
            common_test = np.array([extract_common_features(r) for r in test_recs], dtype=float)

            # ---------------------------------------------------------------
            # Leakage Prevention: Encoders fitted strictly on training fold
            # ---------------------------------------------------------------
            sem_train = None
            sem_test = None
            if mode in ("D", "S+D"):
                # One-hot encode domain inside training fold
                domain_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                train_domains = np.array([
                    [r["semantic_features"].get("probable_domain", "unknown")] for r in train_recs
                ])
                domain_ohe.fit(train_domains)

                # TF-IDF on semantic text descriptions inside training fold
                tfidf = TfidfVectorizer(max_features=25, stop_words="english")
                train_texts = [extract_semantic_text(r) for r in train_recs]
                tfidf.fit(train_texts)

                def _build_sem_features(recs: list[dict[str, Any]]) -> np.ndarray:
                    dom_input = np.array([
                        [r["semantic_features"].get("probable_domain", "unknown")] for r in recs
                    ])
                    dom_vecs = domain_ohe.transform(dom_input)
                    txt_vecs = tfidf.transform([extract_semantic_text(r) for r in recs]).toarray()
                    conf_vecs = np.array([
                        [float(r["semantic_features"].get("domain_confidence") or 0.0)] for r in recs
                    ])
                    return np.hstack([conf_vecs, dom_vecs, txt_vecs])

                sem_train = _build_sem_features(train_recs)
                sem_test = _build_sem_features(test_recs)

            # Assemble mode-specific feature matrices
            if mode == "S":
                struct_train = np.array([extract_structural_features(r) for r in train_recs], dtype=float)
                struct_test = np.array([extract_structural_features(r) for r in test_recs], dtype=float)
                X_train = np.hstack([common_train, struct_train])
                X_test = np.hstack([common_test, struct_test])
            elif mode == "D":
                X_train = np.hstack([common_train, sem_train])
                X_test = np.hstack([common_test, sem_test])
            elif mode == "S+D":
                struct_train = np.array([extract_structural_features(r) for r in train_recs], dtype=float)
                struct_test = np.array([extract_structural_features(r) for r in test_recs], dtype=float)
                X_train = np.hstack([common_train, struct_train, sem_train])
                X_test = np.hstack([common_test, struct_test, sem_test])

            y_train = np.array([r["rank_target"] for r in train_recs], dtype=float)

            # Train simple deterministic meta-model
            meta_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=2,
                random_state=42,
            )
            meta_model.fit(X_train, y_train)

            # Score candidates on held-out dataset
            predicted_scores = meta_model.predict(X_test)

            # Rank candidates descending by predicted relative score
            ranked_indices = np.argsort(-predicted_scores)
            ranked_recommendations = [test_recs[i]["model_name"] for i in ranked_indices]
            recommended_model = ranked_recommendations[0]

            # Ground truth ranking from authoritative benchmark performance
            task_type = test_recs[0]["task_type"]
            actual_sorted = sorted(
                test_recs,
                key=lambda x: x["benchmark_performance"]["primary_metric_value"],
                reverse=(task_type == "classification"),
            )
            actual_ranking = [r["model_name"] for r in actual_sorted]
            actual_best_model = actual_ranking[0]
            actual_best_metric = actual_sorted[0]["benchmark_performance"]["primary_metric_value"]
            actual_worst_metric = actual_sorted[-1]["benchmark_performance"]["primary_metric_value"]

            # Metric of recommended model
            recommended_metric = next(
                r["benchmark_performance"]["primary_metric_value"]
                for r in test_recs
                if r["model_name"] == recommended_model
            )

            # Regret and normalized regret calculations
            if task_type == "classification":
                # Balanced accuracy: higher is better
                regret = actual_best_metric - recommended_metric
                performance_spread = actual_best_metric - actual_worst_metric
            else:
                # RMSE: lower is better
                regret = recommended_metric - actual_best_metric
                performance_spread = actual_worst_metric - actual_best_metric

            normalized_regret = (regret / performance_spread) if performance_spread > 0 else 0.0

            is_top1_correct = bool(recommended_model == actual_best_model)
            is_top2_correct = bool(actual_best_model in ranked_recommendations[:2])

            if is_top1_correct:
                top1_correct_count += 1
            if is_top2_correct:
                top2_correct_count += 1
            all_regrets.append(float(regret))
            all_norm_regrets.append(float(normalized_regret))

            detailed_evaluations[mode].append({
                "dataset_id": holdout_did,
                "dataset_name": test_recs[0]["dataset_name"],
                "task_type": task_type,
                "representation": mode,
                "actual_best_model": actual_best_model,
                "recommended_model": recommended_model,
                "ranked_recommendations": ranked_recommendations,
                "actual_model_ranking": actual_ranking,
                "top1_correct": is_top1_correct,
                "top2_correct": is_top2_correct,
                "regret": float(regret),
                "normalized_regret": float(normalized_regret),
            })

        mean_norm_regret = float(np.mean(all_norm_regrets))
        summary_results[mode] = {
            "datasets_evaluated": num_datasets,
            "top1_accuracy": top1_correct_count,
            "top1_accuracy_pct": float(top1_correct_count / num_datasets * 100.0),
            "top2_accuracy": top2_correct_count,
            "top2_accuracy_pct": float(top2_correct_count / num_datasets * 100.0),
            "mean_regret": float(np.mean(all_regrets)),
            "normalized_regret": mean_norm_regret,
            "oracle_gap": mean_norm_regret,  # Normalized gap relative to Oracle's 0.0 regret
        }

    # 5. Persist evaluation results
    output_payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta_model": "RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=2, random_state=42)",
        "evaluation_protocol": "Leave-One-Dataset-Out (LODO) cross-validation across 51 datasets",
        "primary_metric_specifications": {
            "classification": "balanced_accuracy (higher is better)",
            "regression": "rmse (lower is better)",
        },
        "summary": summary_results,
        "evaluations_by_mode": detailed_evaluations,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    # 6. Compact Terminal Report
    print("\n" + "=" * 70)
    print("LYNX MODEL SELECTOR — LODO BENCHMARK EVALUATION REPORT")
    print("=" * 70)

    for mode in modes:
        res = summary_results[mode]
        print(f"\nMode {mode}:")
        print(f"  Datasets evaluated: {res['datasets_evaluated']}")
        print(f"  Top-1:              {res['top1_accuracy_pct']:.1f}% ({res['top1_accuracy']}/{res['datasets_evaluated']})")
        print(f"  Top-2:              {res['top2_accuracy_pct']:.1f}% ({res['top2_accuracy']}/{res['datasets_evaluated']})")
        print(f"  Mean regret:        {res['mean_regret']:.4f}")
        print(f"  Normalized regret:  {res['normalized_regret']:.4f}")
        print(f"  Oracle gap:         {res['oracle_gap']:.4f}")

    print("\n" + "=" * 70)
    print(f"Detailed results saved to: {OUTPUT_PATH}")
    print("=" * 70)

    return output_payload


if __name__ == "__main__":
    run_lodo_evaluation()

