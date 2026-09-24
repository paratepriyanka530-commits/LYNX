"""Statistical evaluation and paired bootstrap analysis of the LYNX model selector.

This module analyzes the completed Leave-One-Dataset-Out (LODO) evaluation results from
  pipelines/pipeline2_prediction/model_selector/selector_evaluation.json.

It performs:
  1. Dataset-level 95% bootstrap confidence intervals for S, D, and S+D on:
     - Top-1 recommendation accuracy
     - Top-2 recommendation accuracy
     - Mean normalized regret
  2. Simultaneous paired bootstrap difference tests for:
     - D - S
     - (S+D) - S
     - (S+D) - D
     across Top-1, Top-2, and normalized regret.

Bootstrap Protocol:
  - Unit of resampling: datasets (N = 51), NOT individual dataset-model records.
  - Paired resampling: identical resampled dataset indices are used simultaneously
    for both representations in each iteration to preserve correlation structure.
  - Replications: 10,000 bootstrap iterations with a fixed random seed (42).
  - Confidence intervals: 2.5th and 97.5th percentiles of the bootstrap distribution.

Outputs:
  pipelines/pipeline2_prediction/model_selector/selector_statistics.json
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
INPUT_PATH = MODULE_DIR / "selector_evaluation.json"
OUTPUT_PATH = MODULE_DIR / "selector_statistics.json"

BOOTSTRAP_RESAMPLES = 10000
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95


# ---------------------------------------------------------------------------
# Core Statistical Functions
# ---------------------------------------------------------------------------

def calculate_bootstrap_statistics() -> dict[str, Any]:
    """Execute paired dataset-level bootstrap analysis on existing selector evaluation."""
    print("=" * 70)
    print("LYNX MODEL SELECTOR — STATISTICAL BOOTSTRAP EVALUATION")
    print("=" * 70)

    # 1. Load evaluation artifact
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Selector evaluation file not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        eval_payload = json.load(f)

    evals_by_mode = eval_payload.get("evaluations_by_mode", {})
    required_modes = ["S", "D", "S+D"]
    for mode in required_modes:
        if mode not in evals_by_mode:
            raise KeyError(f"Missing required evaluation mode '{mode}' in {INPUT_PATH}")

    s_evals = evals_by_mode["S"]
    d_evals = evals_by_mode["D"]
    sd_evals = evals_by_mode["S+D"]

    num_datasets = len(s_evals)
    if len(d_evals) != num_datasets or len(sd_evals) != num_datasets:
        raise ValueError(
            f"Dataset counts mismatch across modes: S={len(s_evals)}, "
            f"D={len(d_evals)}, S+D={len(sd_evals)}"
        )

    # Verify identical dataset order across all modes
    s_ids = [r["dataset_id"] for r in s_evals]
    d_ids = [r["dataset_id"] for r in d_evals]
    sd_ids = [r["dataset_id"] for r in sd_evals]
    if s_ids != d_ids or s_ids != sd_ids:
        raise ValueError("Evaluations are not aligned by the exact same dataset sequence")

    print(f"Loaded {num_datasets} matched held-out dataset evaluations across S, D, and S+D.")
    print(f"Bootstrap iterations: {BOOTSTRAP_RESAMPLES:,} (seed={RANDOM_SEED})")

    # 2. Extract aligned outcome vectors (length N=51)
    top1_arr = {
        m: np.array([1.0 if r["top1_correct"] else 0.0 for r in evals_by_mode[m]], dtype=float)
        for m in required_modes
    }
    top2_arr = {
        m: np.array([1.0 if r["top2_correct"] else 0.0 for r in evals_by_mode[m]], dtype=float)
        for m in required_modes
    }
    norm_regret_arr = {
        m: np.array([float(r["normalized_regret"]) for r in evals_by_mode[m]], dtype=float)
        for m in required_modes
    }

    # 3. Paired bootstrap resampling across dataset indices
    rng = np.random.default_rng(RANDOM_SEED)
    # Shape: (B, N) -> Draw N dataset indices with replacement B times
    resample_matrix = rng.choice(num_datasets, size=(BOOTSTRAP_RESAMPLES, num_datasets), replace=True)

    # Compute bootstrap realizations for each mode
    # Matrix indexing: top1_arr[m][resample_matrix] has shape (B, N)
    boot_top1 = {m: top1_arr[m][resample_matrix].mean(axis=1) for m in required_modes}
    boot_top2 = {m: top2_arr[m][resample_matrix].mean(axis=1) for m in required_modes}
    boot_norm_regret = {m: norm_regret_arr[m][resample_matrix].mean(axis=1) for m in required_modes}

    # 4. Individual mode confidence intervals (percentile method)
    alpha = 1.0 - CONFIDENCE_LEVEL
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    individual_stats: dict[str, Any] = {}
    for m in required_modes:
        t1_lo, t1_hi = np.percentile(boot_top1[m], [lower_pct, upper_pct])
        t2_lo, t2_hi = np.percentile(boot_top2[m], [lower_pct, upper_pct])
        nr_lo, nr_hi = np.percentile(boot_norm_regret[m], [lower_pct, upper_pct])

        individual_stats[m] = {
            "top1_accuracy": {
                "observed": float(top1_arr[m].mean()),
                "observed_pct": float(top1_arr[m].mean() * 100.0),
                "bootstrap_mean": float(boot_top1[m].mean()),
                "std_error": float(boot_top1[m].std(ddof=1)),
                "ci_95_lower": float(t1_lo),
                "ci_95_upper": float(t1_hi),
                "ci_95_lower_pct": float(t1_lo * 100.0),
                "ci_95_upper_pct": float(t1_hi * 100.0),
            },
            "top2_accuracy": {
                "observed": float(top2_arr[m].mean()),
                "observed_pct": float(top2_arr[m].mean() * 100.0),
                "bootstrap_mean": float(boot_top2[m].mean()),
                "std_error": float(boot_top2[m].std(ddof=1)),
                "ci_95_lower": float(t2_lo),
                "ci_95_upper": float(t2_hi),
                "ci_95_lower_pct": float(t2_lo * 100.0),
                "ci_95_upper_pct": float(t2_hi * 100.0),
            },
            "mean_normalized_regret": {
                "observed": float(norm_regret_arr[m].mean()),
                "bootstrap_mean": float(boot_norm_regret[m].mean()),
                "std_error": float(boot_norm_regret[m].std(ddof=1)),
                "ci_95_lower": float(nr_lo),
                "ci_95_upper": float(nr_hi),
            },
        }

    # 5. Paired bootstrap differences (using identical resample indices)
    comparisons = [
        ("D - S", "D", "S"),
        ("S+D - S", "S+D", "S"),
        ("S+D - D", "S+D", "D"),
    ]
    paired_stats: dict[str, Any] = {}

    for comp_label, mode_a, mode_b in comparisons:
        diff_top1 = boot_top1[mode_a] - boot_top1[mode_b]
        diff_top2 = boot_top2[mode_a] - boot_top2[mode_b]
        diff_norm_regret = boot_norm_regret[mode_a] - boot_norm_regret[mode_b]

        obs_diff_t1 = float((top1_arr[mode_a] - top1_arr[mode_b]).mean())
        obs_diff_t2 = float((top2_arr[mode_a] - top2_arr[mode_b]).mean())
        obs_diff_nr = float((norm_regret_arr[mode_a] - norm_regret_arr[mode_b]).mean())

        t1_lo, t1_hi = np.percentile(diff_top1, [lower_pct, upper_pct])
        t2_lo, t2_hi = np.percentile(diff_top2, [lower_pct, upper_pct])
        nr_lo, nr_hi = np.percentile(diff_norm_regret, [lower_pct, upper_pct])

        # Empirical two-tailed bootstrap p-value
        p_t1 = min(1.0, 2.0 * float(min(np.mean(diff_top1 <= 0.0), np.mean(diff_top1 >= 0.0))))
        p_t2 = min(1.0, 2.0 * float(min(np.mean(diff_top2 <= 0.0), np.mean(diff_top2 >= 0.0))))
        p_nr = min(1.0, 2.0 * float(min(np.mean(diff_norm_regret <= 0.0), np.mean(diff_norm_regret >= 0.0))))

        key_name = f"{mode_a}_minus_{mode_b}".replace("+", "")
        paired_stats[key_name] = {
            "comparison": comp_label,
            "top1_accuracy_diff": {
                "observed_diff": obs_diff_t1,
                "observed_diff_pct": obs_diff_t1 * 100.0,
                "bootstrap_mean_diff": float(diff_top1.mean()),
                "std_error": float(diff_top1.std(ddof=1)),
                "ci_95_lower": float(t1_lo),
                "ci_95_upper": float(t1_hi),
                "ci_95_lower_pct": float(t1_lo * 100.0),
                "ci_95_upper_pct": float(t1_hi * 100.0),
                "prob_superior": float(np.mean(diff_top1 > 0.0)),
                "p_value_two_sided": p_t1,
            },
            "top2_accuracy_diff": {
                "observed_diff": obs_diff_t2,
                "observed_diff_pct": obs_diff_t2 * 100.0,
                "bootstrap_mean_diff": float(diff_top2.mean()),
                "std_error": float(diff_top2.std(ddof=1)),
                "ci_95_lower": float(t2_lo),
                "ci_95_upper": float(t2_hi),
                "ci_95_lower_pct": float(t2_lo * 100.0),
                "ci_95_upper_pct": float(t2_hi * 100.0),
                "prob_superior": float(np.mean(diff_top2 > 0.0)),
                "p_value_two_sided": p_t2,
            },
            "mean_normalized_regret_diff": {
                "observed_diff": obs_diff_nr,
                "bootstrap_mean_diff": float(diff_norm_regret.mean()),
                "std_error": float(diff_norm_regret.std(ddof=1)),
                "ci_95_lower": float(nr_lo),
                "ci_95_upper": float(nr_hi),
                "prob_reduction": float(np.mean(diff_norm_regret < 0.0)),
                "p_value_two_sided": p_nr,
            },
        }

    # 6. Save results to selector_statistics.json
    output_payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_artifact": "selector_evaluation.json",
        "bootstrap_config": {
            "num_datasets": num_datasets,
            "resamples": BOOTSTRAP_RESAMPLES,
            "random_seed": RANDOM_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "method": "paired dataset-level percentile bootstrap",
        },
        "individual_modes": individual_stats,
        "paired_differences": paired_stats,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    # 7. Print comprehensive terminal report
    print("\n" + "=" * 70)
    print("INDIVIDUAL REPRESENTATION PERFORMANCE (95% BOOTSTRAP CI)")
    print("=" * 70)
    for m in required_modes:
        s = individual_stats[m]
        print(f"\nMode {m}:")
        print(
            f"  Top-1 Accuracy:     {s['top1_accuracy']['observed_pct']:5.1f}%  "
            f"[95% CI: {s['top1_accuracy']['ci_95_lower_pct']:5.1f}%, {s['top1_accuracy']['ci_95_upper_pct']:5.1f}%]"
        )
        print(
            f"  Top-2 Accuracy:     {s['top2_accuracy']['observed_pct']:5.1f}%  "
            f"[95% CI: {s['top2_accuracy']['ci_95_lower_pct']:5.1f}%, {s['top2_accuracy']['ci_95_upper_pct']:5.1f}%]"
        )
        print(
            f"  Normalized Regret:  {s['mean_normalized_regret']['observed']:6.4f}  "
            f"[95% CI: {s['mean_normalized_regret']['ci_95_lower']:6.4f}, {s['mean_normalized_regret']['ci_95_upper']:6.4f}]"
        )

    print("\n" + "=" * 70)
    print("PAIRED DIFFERENCE TESTS (SIMULTANEOUS DATASET RESAMPLING)")
    print("=" * 70)
    for comp_label, mode_a, mode_b in comparisons:
        key_name = f"{mode_a}_minus_{mode_b}".replace("+", "")
        p_stat = paired_stats[key_name]
        t1 = p_stat["top1_accuracy_diff"]
        t2 = p_stat["top2_accuracy_diff"]
        nr = p_stat["mean_normalized_regret_diff"]

        print(f"\nComparison: {comp_label}")
        print(
            f"  Delta Top-1:        {t1['observed_diff_pct']:+5.1f}%  "
            f"[95% CI: {t1['ci_95_lower_pct']:+5.1f}%, {t1['ci_95_upper_pct']:+5.1f}%]  (p = {t1['p_value_two_sided']:.4f})"
        )
        print(
            f"  Delta Top-2:        {t2['observed_diff_pct']:+5.1f}%  "
            f"[95% CI: {t2['ci_95_lower_pct']:+5.1f}%, {t2['ci_95_upper_pct']:+5.1f}%]  (p = {t2['p_value_two_sided']:.4f})"
        )
        print(
            f"  Delta Norm Regret:  {nr['observed_diff']:+6.4f}  "
            f"[95% CI: {nr['ci_95_lower']:+6.4f}, {nr['ci_95_upper']:+6.4f}]  (p = {nr['p_value_two_sided']:.4f})"
        )

    print("\n" + "=" * 70)
    print(f"Statistical evaluation saved to: {OUTPUT_PATH}")
    print("=" * 70)

    return output_payload


if __name__ == "__main__":
    calculate_bootstrap_statistics()

