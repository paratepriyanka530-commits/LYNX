"""Evaluate the Feasibility-Filtered Semantic Model Selector (F+D).

This script tests the research hypothesis that structural dataset information is more
effectively utilized as a deterministic model-feasibility filter prior to selection,
rather than as a predictive feature vector concatenated with semantic features (S+D).

Protocol:
  1. Uses the existing Leave-One-Dataset-Out (LODO) evaluation framework and the
     pure semantic (D) model selector predictions from selector_evaluation.json.
  2. For each held-out dataset, applies feasibility rules derived strictly from
     authoritative model capabilities (model_capabilities.json) and frozen project
     constraints before model execution.
  3. Prunes infeasible models from the candidate pool and evaluates the remaining
     candidates ordered by the learned semantic selector.
  4. Compares S, D, S+D, and F+D on Top-1, Top-2, normalized regret, candidate
     filtering rates, and oracle-filter penalty.

Outputs:
  pipelines/pipeline2_prediction/model_selector/feasibility_semantic_evaluation.json
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent
PIPELINE2_DIR = MODULE_DIR.parent
INPUTS_DIR = PIPELINE2_DIR / "inputs"

SELECTOR_EVAL_PATH = MODULE_DIR / "selector_evaluation.json"
SELECTOR_DATASET_PATH = MODULE_DIR / "selector_dataset.json"
MODEL_CAPABILITIES_PATH = INPUTS_DIR / "model_capabilities.json"
OUTPUT_PATH = MODULE_DIR / "feasibility_semantic_evaluation.json"


# ---------------------------------------------------------------------------
# Feasibility Rules
# ---------------------------------------------------------------------------

def evaluate_model_feasibility(
    model_id: str,
    capability: dict[str, Any],
    dataset_info: dict[str, Any],
) -> tuple[bool, str | None]:
    """Determine whether a candidate model is feasible for a dataset before execution.
    
    Feasibility rules are derived conservatively from model_capabilities.json:
      1. Supported Task Type: dataset task_type must be in supported_task_types.
      2. Maximum Training Rows (Hard Limit): dataset n_rows <= max_train_rows
         (e.g., TabPFN 10,000 context window).
      3. Maximum Features (Hard Limit): dataset n_features <= max_features
         (e.g., TabPFN 500 feature bound).
      4. Maximum Classes (Hard Limit): for classification, n_classes <= max_classes
         (e.g., LightGBM/XGBoost/CatBoost 2-class binary wrapper limits; TabPFN 10 classes).
      5. Minimum Training Rows (Hard Limit): dataset n_rows >= min_train_rows
         (e.g., XTab 10 rows; others 2 rows).
    """
    task_type = dataset_info["task_type"]
    supported_tasks = capability.get("supported_task_types", [])
    if task_type not in supported_tasks:
        return False, f"Task type '{task_type}' not in supported_task_types {supported_tasks}"

    limits = capability.get("hard_limits", {})
    n_rows = dataset_info.get("n_rows", 0)
    n_features = dataset_info.get("n_features", 0)
    n_classes = dataset_info.get("n_classes")

    # Minimum rows
    min_rows = limits.get("min_train_rows")
    if min_rows is not None and n_rows < min_rows:
        return False, f"Dataset rows ({n_rows}) below min_train_rows ({min_rows})"

    # Maximum rows (e.g. TabPFN 10,000)
    max_rows = limits.get("max_train_rows")
    if max_rows is not None and n_rows > max_rows:
        return False, f"Dataset rows ({n_rows}) exceeds max_train_rows ({max_rows})"

    # Maximum features (e.g. TabPFN 500)
    max_features = limits.get("max_features")
    if max_features is not None and n_features > max_features:
        return False, f"Dataset features ({n_features}) exceeds max_features ({max_features})"

    # Maximum classes (classification only)
    if task_type == "classification" and n_classes is not None:
        max_classes = limits.get("max_classes")
        if max_classes is not None and n_classes > max_classes:
            return False, f"Class count ({n_classes}) exceeds max_classes ({max_classes})"

    return True, None


# ---------------------------------------------------------------------------
# Main Evaluation Workflow
# ---------------------------------------------------------------------------

def run_feasibility_evaluation() -> dict[str, Any]:
    """Evaluate the F+D selector against S, D, and S+D baseline results."""
    print("=" * 70)
    print("LYNX MODEL SELECTOR — FEASIBILITY + SEMANTIC (F+D) EVALUATION")
    print("=" * 70)

    # 1. Load inputs
    if not SELECTOR_EVAL_PATH.is_file():
        raise FileNotFoundError(f"Selector evaluation artifact missing: {SELECTOR_EVAL_PATH}")
    if not SELECTOR_DATASET_PATH.is_file():
        raise FileNotFoundError(f"Selector dataset artifact missing: {SELECTOR_DATASET_PATH}")
    if not MODEL_CAPABILITIES_PATH.is_file():
        raise FileNotFoundError(f"Model capabilities artifact missing: {MODEL_CAPABILITIES_PATH}")

    with SELECTOR_EVAL_PATH.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with SELECTOR_DATASET_PATH.open("r", encoding="utf-8") as f:
        dataset_data = json.load(f)
    with MODEL_CAPABILITIES_PATH.open("r", encoding="utf-8") as f:
        caps_data = json.load(f)

    capabilities_by_id = {r["model_id"]: r for r in caps_data["records"]}
    d_evaluations = eval_data["evaluations_by_mode"]["D"]

    # Extract dataset structural and task metadata lookup
    datasets_info: dict[str, dict[str, Any]] = {}
    dataset_performance: dict[str, dict[str, float]] = {}

    for r in dataset_data["records"]:
        did = str(r["dataset_id"])
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
        if r.get("success"):
            m_name = r["model_name"]
            val = r["benchmark_performance"]["primary_metric_value"]
            dataset_performance.setdefault(did, {})[m_name] = val

    num_datasets = len(d_evaluations)
    print(f"Loaded {num_datasets} held-out dataset evaluation records.")
    print("Applying authoritative feasibility filter rules before semantic ranking...\n")

    # 2. Evaluate F+D on each held-out dataset
    fd_dataset_records: list[dict[str, Any]] = []
    top1_correct_count = 0
    top2_correct_count = 0
    all_regrets: list[float] = []
    all_norm_regrets: list[float] = []
    oracle_filtered_count = 0
    datasets_with_filtered_candidates = 0
    total_candidates_removed = 0

    for d_eval in d_evaluations:
        did = str(d_eval["dataset_id"])
        ds_info = datasets_info[did]
        d_ranking = d_eval["ranked_recommendations"]
        actual_best_model = d_eval["actual_best_model"]
        actual_model_ranking = d_eval["actual_model_ranking"]
        ds_perf = dataset_performance[did]

        # Determine feasibility of each candidate model in the original D ranking
        feasible_models: list[str] = []
        filtered_models: list[str] = []
        filter_reasons: dict[str, str] = {}

        for model_id in d_ranking:
            cap = capabilities_by_id.get(model_id, {})
            is_feasible, reason = evaluate_model_feasibility(model_id, cap, ds_info)
            if is_feasible:
                feasible_models.append(model_id)
            else:
                filtered_models.append(model_id)
                filter_reasons[model_id] = reason or "Infeasible"

        original_count = len(d_ranking)
        feasible_count = len(feasible_models)
        removed_count = len(filtered_models)

        if removed_count > 0:
            datasets_with_filtered_candidates += 1
            total_candidates_removed += removed_count

        # Check if the oracle (actual benchmark-best) model was filtered
        was_oracle_filtered = actual_best_model in filtered_models
        if was_oracle_filtered:
            oracle_filtered_count += 1

        # Final ranking: preserve D's relative ordering among feasible candidates
        # If all were somehow filtered, fall back to D ranking
        final_ranking = feasible_models if feasible_models else d_ranking
        recommended_model = final_ranking[0]

        # Metric evaluations
        task_type = ds_info["task_type"]
        best_metric_val = ds_perf[actual_best_model]
        worst_metric_val = ds_perf[actual_model_ranking[-1]]
        rec_metric_val = ds_perf[recommended_model]

        if task_type == "classification":
            regret = best_metric_val - rec_metric_val
            spread = best_metric_val - worst_metric_val
        else:
            regret = rec_metric_val - best_metric_val
            spread = worst_metric_val - best_metric_val

        norm_regret = (regret / spread) if spread > 0 else 0.0

        is_top1 = bool(recommended_model == actual_best_model)
        is_top2 = bool(actual_best_model in final_ranking[:2])

        if is_top1:
            top1_correct_count += 1
        if is_top2:
            top2_correct_count += 1
        all_regrets.append(float(regret))
        all_norm_regrets.append(float(norm_regret))

        fd_dataset_records.append({
            "dataset_id": d_eval["dataset_id"],
            "dataset_name": d_eval["dataset_name"],
            "task_type": task_type,
            "original_candidate_count": original_count,
            "feasible_candidate_count": feasible_count,
            "filtered_models": filtered_models,
            "reason_each_model_filtered": filter_reasons,
            "actual_benchmark_best_model": actual_best_model,
            "whether_oracle_successful_model_filtered": was_oracle_filtered,
            "d_ranking_before_filtering": d_ranking,
            "final_ranking_after_filtering": final_ranking,
            "top1_correct": is_top1,
            "top2_correct": is_top2,
            "regret": float(regret),
            "normalized_regret": float(norm_regret),
        })

    # 3. Aggregate metrics for F+D
    fd_summary = {
        "datasets_evaluated": num_datasets,
        "top1_accuracy": top1_correct_count,
        "top1_accuracy_pct": float(top1_correct_count / num_datasets * 100.0),
        "top2_accuracy": top2_correct_count,
        "top2_accuracy_pct": float(top2_correct_count / num_datasets * 100.0),
        "mean_regret": float(np.mean(all_regrets)),
        "mean_normalized_regret": float(np.mean(all_norm_regrets)),
        "percentage_of_datasets_with_filtered_candidate": float(
            datasets_with_filtered_candidates / num_datasets * 100.0
        ),
        "mean_number_of_candidates_removed": float(total_candidates_removed / num_datasets),
        "oracle_filtered_count": oracle_filtered_count,
        "oracle_filter_rate": float(oracle_filtered_count / num_datasets * 100.0),
    }

    # Extract existing S, D, S+D summaries from selector_evaluation.json
    s_summary = eval_data["summary"]["S"]
    d_summary = eval_data["summary"]["D"]
    sd_summary = eval_data["summary"]["S+D"]

    comparison_summary = {
        "S": {
            "top1_accuracy_pct": s_summary["top1_accuracy_pct"],
            "top2_accuracy_pct": s_summary["top2_accuracy_pct"],
            "mean_normalized_regret": s_summary["normalized_regret"],
        },
        "D": {
            "top1_accuracy_pct": d_summary["top1_accuracy_pct"],
            "top2_accuracy_pct": d_summary["top2_accuracy_pct"],
            "mean_normalized_regret": d_summary["normalized_regret"],
        },
        "S+D": {
            "top1_accuracy_pct": sd_summary["top1_accuracy_pct"],
            "top2_accuracy_pct": sd_summary["top2_accuracy_pct"],
            "mean_normalized_regret": sd_summary["normalized_regret"],
        },
        "F+D": {
            "top1_accuracy_pct": fd_summary["top1_accuracy_pct"],
            "top2_accuracy_pct": fd_summary["top2_accuracy_pct"],
            "mean_normalized_regret": fd_summary["mean_normalized_regret"],
            "percentage_datasets_filtered": fd_summary["percentage_of_datasets_with_filtered_candidate"],
            "mean_candidates_removed": fd_summary["mean_number_of_candidates_removed"],
            "oracle_filter_rate": fd_summary["oracle_filter_rate"],
        },
    }

    # 4. Save results to feasibility_semantic_evaluation.json
    output_payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hypothesis": (
            "Structural information is more useful as a model-feasibility filter (F+D) "
            "than as a predictive feature concatenated with semantic features (S+D)."
        ),
        "evaluation_protocol": "Leave-One-Dataset-Out (LODO) across 51 datasets",
        "feasibility_rule_source": "pipelines/pipeline2_prediction/inputs/model_capabilities.json",
        "comparison_summary": comparison_summary,
        "fd_summary": fd_summary,
        "evaluations_by_dataset": fd_dataset_records,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    # 5. Print terminal report
    print("\n" + "=" * 70)
    print("COMPARATIVE EVALUATION: S vs. D vs. S+D vs. F+D")
    print("=" * 70)
    print(f"{'Mode':<10} | {'Top-1 Acc':<12} | {'Top-2 Acc':<12} | {'Mean Norm Regret':<18}")
    print("-" * 70)
    print(f"{'S':<10} | {s_summary['top1_accuracy_pct']:>5.1f}%       | {s_summary['top2_accuracy_pct']:>5.1f}%       | {s_summary['normalized_regret']:>10.4f}")
    print(f"{'D':<10} | {d_summary['top1_accuracy_pct']:>5.1f}%       | {d_summary['top2_accuracy_pct']:>5.1f}%       | {d_summary['normalized_regret']:>10.4f}")
    print(f"{'S+D':<10} | {sd_summary['top1_accuracy_pct']:>5.1f}%       | {sd_summary['top2_accuracy_pct']:>5.1f}%       | {sd_summary['normalized_regret']:>10.4f}")
    print(f"{'F+D':<10} | {fd_summary['top1_accuracy_pct']:>5.1f}%       | {fd_summary['top2_accuracy_pct']:>5.1f}%       | {fd_summary['mean_normalized_regret']:>10.4f}")
    print("-" * 70)

    print("\nFeasibility Filter Diagnostics (F+D):")
    print(f"  Percentage of datasets with >= 1 candidate filtered: {fd_summary['percentage_of_datasets_with_filtered_candidate']:.1f}% ({datasets_with_filtered_candidates}/{num_datasets})")
    print(f"  Mean number of candidates removed per dataset:        {fd_summary['mean_number_of_candidates_removed']:.2f}")
    print(f"  Oracle-filter rate (best model erroneously pruned):  {fd_summary['oracle_filter_rate']:.1f}% ({oracle_filtered_count}/{num_datasets})")
    print("=" * 70)
    print(f"Detailed results saved to: {OUTPUT_PATH}")
    print("=" * 70)

    return output_payload


if __name__ == "__main__":
    run_feasibility_evaluation()

