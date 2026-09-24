"""Train and persist the final production LYNX meta-level model selector.

This module trains the final learned model selector for the LYNX AutoML system.
Configuration:
  - Representation Mode: Validated semantic representation (D).
  - Features:
      * Shared Model Identity & Family (One-Hot)
      * Shared Model Capabilities (missing, categorical, probability, GPU flags)
      * Shared Task & Target Metadata (task type, classification type, classes, imbalance)
      * Semantic Dataset Features (Domain One-Hot, Domain Confidence, Text TF-IDF)
  - Target:
      * Normalized relative rank target in [0, 1] derived strictly from 51x10 benchmark performance:
          - Classification: balanced_accuracy (higher is better)
          - Regression: rmse (lower is better)
          - Empirical best model receives 1.0; worst receives 0.0.
  - Model:
      * RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=2, random_state=42)
  - Data:
      * Fitted on all 437 successful dataset-model observations across all 51 benchmark datasets.
      * Excludes 73 benchmark failures from supervised training (no zero-score imputation).
      * Zero exposure of benchmark performance metrics or runtimes as predictor features.

Outputs:
  - pipelines/pipeline2_prediction/model_selector/lynx_model_selector.joblib
  - pipelines/pipeline2_prediction/model_selector/final_selector_metadata.json
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import joblib
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
SELECTOR_STATS_PATH = MODULE_DIR / "selector_statistics.json"
MODEL_CAPABILITIES_PATH = INPUTS_DIR / "model_capabilities.json"

MODEL_ARTIFACT_PATH = MODULE_DIR / "lynx_model_selector.joblib"
METADATA_PATH = MODULE_DIR / "final_selector_metadata.json"

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

def extract_common_features_from_dict(
    model_name: str,
    capability: dict[str, Any],
    task_type: str,
    target_info: dict[str, Any],
) -> list[float]:
    """Extract model identity, capability, and task/target metadata vector (29 features)."""
    model_onehot = [1.0 if model_name == m else 0.0 for m in CANONICAL_MODELS]

    fam = capability.get("model_family", "")
    family_onehot = [1.0 if fam == f else 0.0 for f in CANONICAL_FAMILIES]

    caps = capability.get("capabilities", {})
    h_miss = 1.0 if caps.get("handles_missing_natively") is True else (
        0.0 if caps.get("handles_missing_natively") is False else 0.5
    )
    h_cat = 1.0 if caps.get("handles_categorical_natively") is True else (
        0.0 if caps.get("handles_categorical_natively") is False else 0.5
    )
    s_prob = 1.0 if caps.get("supports_probabilities") else 0.0

    prov = capability.get("runtime_and_provisioning", {})
    gpu_req = 1.0 if prov.get("gpu_required") else 0.0
    gpu_opt = 1.0 if prov.get("gpu_optional") else 0.0

    is_cls = 1.0 if task_type == "classification" else 0.0
    ctype = target_info.get("classification_type")
    is_bin = 1.0 if ctype == "binary" else 0.0
    is_multi = 1.0 if ctype == "multiclass" else 0.0
    n_cls = float(target_info.get("n_classes") or 1)
    min_ratio = float(target_info.get("minority_class_ratio") or 0.0)
    missing_pct = float(target_info.get("target_missing_pct") or 0.0)
    cb_chl = 1.0 if target_info.get("cb_chl_eligible") else 0.0

    return (
        model_onehot
        + family_onehot
        + [h_miss, h_cat, s_prob, gpu_req, gpu_opt]
        + [is_cls, is_bin, is_multi, n_cls, min_ratio, missing_pct, cb_chl]
    )


def extract_semantic_text(semantic_features: dict[str, Any]) -> str:
    """Concatenate semantic textual descriptions for TF-IDF vectorization."""
    tokens = [
        str(semantic_features.get("probable_domain") or ""),
        str(semantic_features.get("subdomain") or ""),
        str(semantic_features.get("dataset_purpose") or ""),
        str(semantic_features.get("row_entity") or ""),
        str(semantic_features.get("target_concept") or ""),
        str(semantic_features.get("feature_context") or ""),
        str(semantic_features.get("semantic_summary") or ""),
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
# Production Model Selector Class
# ---------------------------------------------------------------------------

class LYNXModelSelector:
    """Production inference interface for the LYNX meta-level model selector."""

    def __init__(self, bundle: dict[str, Any]):
        self.meta_model: RandomForestRegressor = bundle["meta_model"]
        self.domain_encoder: OneHotEncoder = bundle["domain_encoder"]
        self.text_vectorizer: TfidfVectorizer = bundle["text_vectorizer"]
        self.canonical_models: list[str] = bundle["canonical_models"]
        self.canonical_families: list[str] = bundle["canonical_families"]
        self.capabilities_by_model: dict[str, dict[str, Any]] = bundle["capabilities_by_model"]
        self.representation: str = bundle["representation"]
        self.metadata: dict[str, Any] = bundle["metadata"]

    @classmethod
    def load(cls, path: Path | str) -> LYNXModelSelector:
        """Load a persisted model selector bundle from disk."""
        bundle = joblib.load(path)
        return cls(bundle)

    def predict_ranking(
        self,
        task_type: str,
        target_info: dict[str, Any] | None = None,
        semantic_features: dict[str, Any] | None = None,
        candidate_subset: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Score and rank candidate models for an unseen dataset without benchmark outcomes.
        
        Args:
            task_type: "classification" or "regression".
            target_info: Target properties (n_classes, classification_type, etc.).
            semantic_features: Holistic semantic profile (probable_domain, semantic_summary, etc.).
            candidate_subset: Optional list of specific models to evaluate (defaults to all 10).
            
        Returns:
            List of candidate model recommendation records sorted by predicted suitability.
        """
        tgt = target_info or {}
        sem = semantic_features or {}
        candidates = candidate_subset or self.canonical_models

        # Semantic feature transformations
        domain_name = sem.get("probable_domain", "unknown")
        dom_vec = self.domain_encoder.transform([[domain_name]])
        text_str = extract_semantic_text(sem)
        txt_vec = self.text_vectorizer.transform([text_str]).toarray()
        conf_vec = np.array([[float(sem.get("domain_confidence") or 0.0)]])
        sem_vector = np.hstack([conf_vec, dom_vec, txt_vec])  # Shape: (1, 51)

        # Build feature row for each candidate model
        feature_rows: list[np.ndarray] = []
        for model_name in candidates:
            cap = self.capabilities_by_model.get(model_name, {})
            common_vec = np.array([
                extract_common_features_from_dict(model_name, cap, task_type, tgt)
            ], dtype=float)
            row = np.hstack([common_vec, sem_vector])
            feature_rows.append(row)

        X = np.vstack(feature_rows)  # Shape: (len(candidates), 80)
        predicted_scores = self.meta_model.predict(X)

        ranked_indices = np.argsort(-predicted_scores)
        ranked_records = []
        for rank_idx, i in enumerate(ranked_indices, start=1):
            m_name = candidates[i]
            cap = self.capabilities_by_model.get(m_name, {})
            ranked_records.append({
                "rank": rank_idx,
                "model_name": m_name,
                "model_family": cap.get("model_family", "unknown"),
                "predicted_suitability_score": float(predicted_scores[i]),
            })

        return ranked_records


# ---------------------------------------------------------------------------
# Training and Persistence
# ---------------------------------------------------------------------------

def train_and_persist_final_selector() -> tuple[Path, Path]:
    """Train the final LYNX model selector on all 51 benchmark datasets and persist artifacts."""
    print("=" * 75)
    print("LYNX MODEL SELECTOR — FINAL PRODUCTION TRAINING (MODE D)")
    print("=" * 75)

    # 1. Load inputs
    if not SELECTOR_DATASET_PATH.is_file():
        raise FileNotFoundError(f"Selector dataset not found: {SELECTOR_DATASET_PATH}")
    if not MODEL_CAPABILITIES_PATH.is_file():
        raise FileNotFoundError(f"Model capabilities not found: {MODEL_CAPABILITIES_PATH}")
    if not SELECTOR_EVAL_PATH.is_file():
        raise FileNotFoundError(f"Selector evaluation not found: {SELECTOR_EVAL_PATH}")
    if not SELECTOR_STATS_PATH.is_file():
        raise FileNotFoundError(f"Selector statistics not found: {SELECTOR_STATS_PATH}")

    with SELECTOR_DATASET_PATH.open("r", encoding="utf-8") as f:
        dataset_json = json.load(f)
    with MODEL_CAPABILITIES_PATH.open("r", encoding="utf-8") as f:
        caps_json = json.load(f)
    with SELECTOR_EVAL_PATH.open("r", encoding="utf-8") as f:
        eval_json = json.load(f)
    with SELECTOR_STATS_PATH.open("r", encoding="utf-8") as f:
        stats_json = json.load(f)

    all_records = dataset_json.get("records", [])
    successful_records = [r for r in all_records if r.get("success") is True]
    failed_records = [r for r in all_records if r.get("success") is not True]

    print(f"Total benchmark records:    {len(all_records)}")
    print(f"Successful records trained: {len(successful_records)}")
    print(f"Failed records excluded:    {len(failed_records)}")

    # 2. Group records by dataset ID and compute ranking targets
    records_by_dataset: dict[Any, list[dict[str, Any]]] = {}
    for r in successful_records:
        records_by_dataset.setdefault(r["dataset_id"], []).append(r)

    dataset_ids = sorted(list(records_by_dataset.keys()))
    num_datasets = len(dataset_ids)
    print(f"Unique source datasets:     {num_datasets}")

    compute_dataset_ranking_targets(records_by_dataset)

    # 3. Fit preprocessing components on the complete dataset
    print("\n[Phase 1] Fitting encoders on full training corpus (437 observations)...")
    capabilities_by_id = {r["model_id"]: r for r in caps_json["records"]}

    # Domain One-Hot Encoder
    domain_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_domains = np.array([
        [r["semantic_features"].get("probable_domain", "unknown")]
        for r in successful_records
    ])
    domain_ohe.fit(train_domains)
    print(f"  Fitted Domain OHE: {len(domain_ohe.categories_[0])} unique categories")

    # Text TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=25, stop_words="english")
    train_texts = [extract_semantic_text(r["semantic_features"]) for r in successful_records]
    tfidf.fit(train_texts)
    print(f"  Fitted TF-IDF Vectorizer: {len(tfidf.get_feature_names_out())} vocabulary terms")

    # 4. Construct feature matrix and target vector
    print("\n[Phase 2] Constructing feature matrix (Mode D)...")
    common_matrix = np.array([
        extract_common_features_from_dict(
            r["model_name"],
            r.get("model_capabilities", {}),
            r["task_type"],
            r.get("target_info", {}),
        )
        for r in successful_records
    ], dtype=float)

    dom_features = domain_ohe.transform(train_domains)
    txt_features = tfidf.transform(train_texts).toarray()
    conf_features = np.array([
        [float(r["semantic_features"].get("domain_confidence") or 0.0)]
        for r in successful_records
    ], dtype=float)

    sem_matrix = np.hstack([conf_features, dom_features, txt_features])
    X_train = np.hstack([common_matrix, sem_matrix])
    y_train = np.array([r["rank_target"] for r in successful_records], dtype=float)

    print(f"  X_train shape: {X_train.shape} (29 common + 51 semantic = 80 features)")
    print(f"  y_train shape: {y_train.shape} (normalized rank targets in [0, 1])")

    # 5. Fit the validated RandomForestRegressor configuration
    print("\n[Phase 3] Fitting final RandomForestRegressor meta-model...")
    meta_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
    )
    meta_model.fit(X_train, y_train)
    print(f"  RandomForestRegressor fit complete. Score on training set: {meta_model.score(X_train, y_train):.4f}")

    # 6. Package bundle and persist lynx_model_selector.joblib
    print("\n[Phase 4] Packaging artifact bundle...")
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    bundle = {
        "schema_version": "1.0",
        "created_at": created_at,
        "selector_name": "LYNX Meta-Level Model Selector",
        "representation": "D",
        "meta_model": meta_model,
        "domain_encoder": domain_ohe,
        "text_vectorizer": tfidf,
        "canonical_models": CANONICAL_MODELS,
        "canonical_families": CANONICAL_FAMILIES,
        "capabilities_by_model": capabilities_by_id,
        "expected_feature_dimensions": {
            "common_features": common_matrix.shape[1],
            "semantic_features": sem_matrix.shape[1],
            "total_features": X_train.shape[1],
        },
        "target_specification": {
            "classification_primary_metric": "balanced_accuracy",
            "regression_primary_metric": "rmse",
            "target_scale": "[0.0, 1.0]",
            "rule": "1.0 corresponds to the empirical best model on each dataset",
        },
        "metadata": {
            "source_datasets": num_datasets,
            "training_records": len(successful_records),
            "excluded_failures": len(failed_records),
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
        },
    }

    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_ARTIFACT_PATH, compress=3)
    file_size_kb = MODEL_ARTIFACT_PATH.stat().st_size / 1024
    print(f"  Artifact saved to: {MODEL_ARTIFACT_PATH} ({file_size_kb:.1f} KB)")

    # 7. Generate final_selector_metadata.json
    print("\n[Phase 5] Writing comprehensive metadata artifact...")
    d_lodo_summary = eval_json["summary"]["D"]
    d_stats = stats_json["individual_modes"]["D"]
    d_minus_s = stats_json["paired_differences"]["D_minus_S"]
    sd_minus_d = stats_json["paired_differences"]["SD_minus_D"]

    metadata_payload = {
        "schema_version": "1.0",
        "created_at": created_at,
        "selector_name": "LYNX Meta-Level Model Selector",
        "representation": "D",
        "meta_model_type": "RandomForestRegressor",
        "number_of_source_benchmark_datasets": num_datasets,
        "number_of_successful_dataset_model_training_observations": len(successful_records),
        "number_of_failed_benchmark_observations_excluded_from_supervised_training": len(failed_records),
        "candidate_models": CANONICAL_MODELS,
        "candidate_model_families": CANONICAL_FAMILIES,
        "classification_primary_metric": "balanced_accuracy",
        "regression_primary_metric": "rmse",
        "normalized_rank_target_definition": (
            "Within each dataset, signed primary metric values (balanced_accuracy for classification, "
            "-rmse for regression) are converted to ranks in [0, 1] using (rank - 1)/(k - 1), "
            "where the empirical best model receives 1.0 and the worst receives 0.0."
        ),
        "final_model_hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_leaf": 2,
            "random_state": 42,
        },
        "feature_schema": {
            "common_features_dim": 29,
            "semantic_features_dim": 51,
            "total_features_dim": 80,
            "semantic_feature_breakdown": {
                "domain_confidence": 1,
                "domain_one_hot": len(domain_ohe.categories_[0]),
                "tfidf_semantic_tokens": len(tfidf.get_feature_names_out()),
            },
        },
        "frozen_lodo_results_for_D": {
            "datasets_evaluated": d_lodo_summary["datasets_evaluated"],
            "top1_accuracy": d_lodo_summary["top1_accuracy"],
            "top1_accuracy_pct": d_lodo_summary["top1_accuracy_pct"],
            "top2_accuracy": d_lodo_summary["top2_accuracy"],
            "top2_accuracy_pct": d_lodo_summary["top2_accuracy_pct"],
            "mean_regret": d_lodo_summary["mean_regret"],
            "mean_normalized_regret": d_lodo_summary["normalized_regret"],
            "oracle_gap": d_lodo_summary["oracle_gap"],
        },
        "relevant_bootstrap_results": {
            "confidence_level": stats_json["bootstrap_config"]["confidence_level"],
            "resamples": stats_json["bootstrap_config"]["resamples"],
            "mode_D_estimates": {
                "top1_accuracy_observed_pct": d_stats["top1_accuracy"]["observed_pct"],
                "top1_accuracy_95_ci_pct": [
                    d_stats["top1_accuracy"]["ci_95_lower_pct"],
                    d_stats["top1_accuracy"]["ci_95_upper_pct"],
                ],
                "top2_accuracy_observed_pct": d_stats["top2_accuracy"]["observed_pct"],
                "top2_accuracy_95_ci_pct": [
                    d_stats["top2_accuracy"]["ci_95_lower_pct"],
                    d_stats["top2_accuracy"]["ci_95_upper_pct"],
                ],
                "mean_normalized_regret_observed": d_stats["mean_normalized_regret"]["observed"],
                "mean_normalized_regret_95_ci": [
                    d_stats["mean_normalized_regret"]["ci_95_lower"],
                    d_stats["mean_normalized_regret"]["ci_95_upper"],
                ],
            },
            "paired_differences": {
                "D_minus_S": {
                    "top1_diff_pct": d_minus_s["top1_accuracy_diff"]["observed_diff_pct"],
                    "top1_diff_95_ci_pct": [
                        d_minus_s["top1_accuracy_diff"]["ci_95_lower_pct"],
                        d_minus_s["top1_accuracy_diff"]["ci_95_upper_pct"],
                    ],
                    "top1_p_value": d_minus_s["top1_accuracy_diff"]["p_value_two_sided"],
                    "top2_diff_pct": d_minus_s["top2_accuracy_diff"]["observed_diff_pct"],
                    "top2_diff_95_ci_pct": [
                        d_minus_s["top2_accuracy_diff"]["ci_95_lower_pct"],
                        d_minus_s["top2_accuracy_diff"]["ci_95_upper_pct"],
                    ],
                    "top2_p_value": d_minus_s["top2_accuracy_diff"]["p_value_two_sided"],
                    "regret_diff": d_minus_s["mean_normalized_regret_diff"]["observed_diff"],
                    "regret_p_value": d_minus_s["mean_normalized_regret_diff"]["p_value_two_sided"],
                },
                "SD_minus_D": {
                    "top1_diff_pct": sd_minus_d["top1_accuracy_diff"]["observed_diff_pct"],
                    "top2_diff_pct": sd_minus_d["top2_accuracy_diff"]["observed_diff_pct"],
                    "regret_diff": sd_minus_d["mean_normalized_regret_diff"]["observed_diff"],
                    "regret_p_value": sd_minus_d["mean_normalized_regret_diff"]["p_value_two_sided"],
                },
            },
        },
        "artifact_file": MODEL_ARTIFACT_PATH.name,
    }

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved to: {METADATA_PATH}")

    # 8. Verification in a clean subprocess
    print("\n[Phase 6] Verifying artifact reload and inference in a fresh Python process...")
    verify_code = f"""
import json
import sys
from pathlib import Path
import joblib

sys.path.insert(0, r'{MODULE_DIR}')
from train_final_selector import LYNXModelSelector

artifact_path = Path(r'{MODEL_ARTIFACT_PATH}')
selector = LYNXModelSelector.load(artifact_path)

# Sample unseen dataset input (strictly pre-execution information only; NO benchmark outcomes)
sample_task_type = "classification"
sample_target_info = {{
    "classification_type": "binary",
    "n_classes": 2,
    "minority_class_ratio": 0.45,
    "target_missing_pct": 0.0,
    "cb_chl_eligible": True,
}}
sample_semantic_profile = {{
    "probable_domain": "finance",
    "subdomain": "credit risk",
    "dataset_purpose": "default prediction",
    "row_entity": "loan applicant",
    "target_concept": "loan default",
    "feature_context": ["credit history", "annual income", "debt ratio"],
    "semantic_summary": "Financial dataset predicting loan default from applicant debt and credit attributes.",
    "domain_confidence": 0.95,
}}

ranking = selector.predict_ranking(
    task_type=sample_task_type,
    target_info=sample_target_info,
    semantic_features=sample_semantic_profile,
)

assert len(ranking) == 10, f"Expected 10 candidates ranked, got {{len(ranking)}}"
print("SUCCESS: Fresh process inference verified.")
print(f"Top-1 Recommendation: {{ranking[0]['model_name']}} (Score: {{ranking[0]['predicted_suitability_score']:.4f}})")
print(f"Top-2 Recommendation: {{ranking[1]['model_name']}} (Score: {{ranking[1]['predicted_suitability_score']:.4f}})")
print("All 10 candidates scored and ranked without benchmark performance fields.")
"""

    result = subprocess.run(
        [sys.executable, "-c", verify_code],
        capture_output=True,
        text=True,
        check=True,
    )
    print(result.stdout.strip())
    print("\n" + "=" * 75)
    print("LYNX MODEL SELECTOR PRODUCTION TRAINING COMPLETE")
    print("=" * 75)

    return MODEL_ARTIFACT_PATH, METADATA_PATH


if __name__ == "__main__":
    train_and_persist_final_selector()

