"""Generate Publication-Quality Figure 2 for IEEE Research Paper.

Figure Title:
  Distribution of Best-Performing Model Families Across 51 Benchmark Datasets

Source:
  pipelines/pipeline2_prediction/training_data/model_performance.json

Protocol:
  1. Reads model_performance.json.
  2. For each of the 51 datasets, selects the best-performing successful model:
       - Classification: balanced_accuracy (higher is better)
       - Regression: rmse (lower is better)
  3. Ignores failed/infeasible runs when determining the winner.
  4. Counts dataset wins for all 10 canonical model families.
  5. Formally asserts derived counts against expected benchmark evidence:
       TabPFN (14), TabICLv2 (13), XGBoost (5), Random Forest (5),
       CatBoost (4), LightGBM (3), FT-Transformer (3), XTab (2), TabM (1), RealMLP (1).
  6. Renders a publication-grade horizontal bar chart (PNG at 600 DPI and PDF vector).

Outputs:
  - fig2_model_winner_distribution.png
  - fig2_model_winner_distribution.pdf
  - pipelines/pipeline2_prediction/model_selector/fig2_model_winner_distribution.png
  - pipelines/pipeline2_prediction/model_selector/fig2_model_winner_distribution.pdf
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent.parent.parent
PERFORMANCE_PATH = MODULE_DIR.parent / "training_data" / "model_performance.json"

PNG_OUTPUT_ROOT = REPO_ROOT / "fig2_model_winner_distribution.png"
PDF_OUTPUT_ROOT = REPO_ROOT / "fig2_model_winner_distribution.pdf"
PNG_OUTPUT_MODULE = MODULE_DIR / "fig2_model_winner_distribution.png"
PDF_OUTPUT_MODULE = MODULE_DIR / "fig2_model_winner_distribution.pdf"

MODEL_DISPLAY_NAMES = {
    "tabpfn": "TabPFN",
    "tabiclv2": "TabICLv2",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "ft_transformer": "FT-Transformer",
    "xtab": "XTab",
    "tabm": "TabM",
    "realmlp": "RealMLP",
}

EXPECTED_WINNER_COUNTS = {
    "TabPFN": 14,
    "TabICLv2": 13,
    "XGBoost": 5,
    "Random Forest": 5,
    "CatBoost": 4,
    "LightGBM": 3,
    "FT-Transformer": 3,
    "XTab": 2,
    "TabM": 1,
    "RealMLP": 1,
}


# ---------------------------------------------------------------------------
# Data Derivation & Verification
# ---------------------------------------------------------------------------

def derive_and_verify_winner_counts() -> tuple[list[str], list[int]]:
    """Derive winner counts directly from model_performance.json and assert equality."""
    if not PERFORMANCE_PATH.is_file():
        raise FileNotFoundError(f"Performance data file missing: {PERFORMANCE_PATH}")

    with PERFORMANCE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_dataset[str(r["dataset_id"])].append(r)

    num_datasets = len(by_dataset)
    if num_datasets != 51:
        raise AssertionError(f"Expected 51 benchmark datasets, found {num_datasets}")

    winners: Counter[str] = Counter()

    for did, recs in by_dataset.items():
        # Exclude failed or infeasible runs
        successful = [
            r for r in recs
            if r.get("primary_metric_value") is not None and r.get("infeasible_reason") is None
        ]
        if not successful:
            raise RuntimeError(f"Dataset {did} contains zero successful benchmark evaluations!")

        task_type = successful[0]["task_type"]
        if task_type == "classification":
            best_rec = max(successful, key=lambda x: x["primary_metric_value"])
        elif task_type == "regression":
            best_rec = min(successful, key=lambda x: x["primary_metric_value"])
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        model_key = best_rec["model_id"]
        display_name = MODEL_DISPLAY_NAMES[model_key]
        winners[display_name] += 1

    # Print terminal verification output
    print(f"Total datasets = {num_datasets}")
    print("Winner count for every model:")
    ordered_models = list(EXPECTED_WINNER_COUNTS.keys())
    derived_counts = []

    for model_name in ordered_models:
        count = winners[model_name]
        derived_counts.append(count)
        expected = EXPECTED_WINNER_COUNTS[model_name]
        print(f"  {model_name:<16}: {count}")
        if count != expected:
            raise AssertionError(
                f"DISCREPANCY DETECTED for {model_name}: expected {expected}, derived {count}. Aborting plot."
            )

    sum_wins = sum(derived_counts)
    print(f"Sum of winner counts = {sum_wins}")
    if sum_wins != 51:
        raise AssertionError(f"Sum of winner counts ({sum_wins}) does not equal 51!")

    return ordered_models, derived_counts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_ieee_figure(ordered_models: list[str], counts: list[int]) -> tuple[Path, Path]:
    """Render publication-quality IEEE horizontal bar chart."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#444444"
    plt.rcParams["axes.linewidth"] = 0.75
    plt.rcParams["text.color"] = "#111111"

    fig, ax = plt.subplots(figsize=(3.8, 3.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = range(len(ordered_models))
    bars = ax.barh(y_pos, counts, height=0.62, color="#2b5c8f", edgecolor="none", zorder=3)

    # Invert Y-axis so TabPFN is at top
    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_models, fontsize=8.5, color="#222222")

    # Bar labels at bar ends
    ax.bar_label(bars, padding=3.5, fontsize=8.5, color="#222222")

    # X-axis configuration
    ax.set_xlim(0, 16)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.tick_params(axis="x", which="both", length=3, width=0.75, color="#555555", labelsize=8.5)
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_xlabel("Number of Benchmark Datasets Won", fontsize=9.0, color="#222222", labelpad=5)

    # Thin light x-grid behind bars
    ax.grid(axis="x", linestyle="--", linewidth=0.5, color="#e2e2e2", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    # Border spines (remove top and right)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    # Save PNG at 600 DPI and PDF vector version
    fig.savefig(PNG_OUTPUT_ROOT, dpi=600, bbox_inches="tight")
    fig.savefig(PDF_OUTPUT_ROOT, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Also keep a copy in the model_selector directory
    shutil.copyfile(PNG_OUTPUT_ROOT, PNG_OUTPUT_MODULE)
    shutil.copyfile(PDF_OUTPUT_ROOT, PDF_OUTPUT_MODULE)

    print("Paths of the saved figures:")
    print(f"  PNG (600 DPI): {PNG_OUTPUT_ROOT}")
    print(f"  PDF (vector):  {PDF_OUTPUT_ROOT}")
    print(f"  Module copy:   {PNG_OUTPUT_MODULE}")
    print(f"                 {PDF_OUTPUT_MODULE}")

    return PNG_OUTPUT_ROOT, PDF_OUTPUT_ROOT


def main() -> None:
    ordered_models, counts = derive_and_verify_winner_counts()
    generate_ieee_figure(ordered_models, counts)


if __name__ == "__main__":
    main()

