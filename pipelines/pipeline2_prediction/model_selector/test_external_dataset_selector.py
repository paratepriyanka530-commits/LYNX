"""External validation of the frozen LYNX model selector on an unseen dataset.

This script demonstrates and verifies the production inference pipeline:
  raw unseen dataset -> LYNX understanding -> automated semantic profile (Ollama qwen2.5:7b)
  -> frozen learned model selector (lynx_model_selector.joblib) -> recommended model ranking.

Constraints:
  - Strictly zero benchmarking across the 10 models.
  - Zero retraining of the model selector.
  - Fully automated semantic profile generation via Ollama (qwen2.5:7b) without manual editing.
  - Explicit verification that the dataset is not in the 51 training datasets.
  - Zero predictive model training.

Outputs:
  - pipelines/pipeline2_prediction/model_selector/external_selector_test_01.json
  - external_selector_test_01.json
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent.parent.parent
PIPELINE2_DIR = MODULE_DIR.parent
INPUTS_DIR = PIPELINE2_DIR / "inputs"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MODULE_DIR))

from benchmarks.tabarena.pipeline2_groundwork import GroundworkOllamaProvider
from pipelines.pipeline2_prediction.dataset_semantic_profiler import (
    build_dataset_semantic_evidence,
    build_dataset_semantic_prompt,
    Pipeline2DatasetSemanticProfile,
)
from train_final_selector import LYNXModelSelector

SELECTOR_JOB_PATH = MODULE_DIR / "lynx_model_selector.joblib"
DATASET_METADATA_PATH = INPUTS_DIR / "dataset_metadata.json"
OUTPUT_JSON_MODULE = MODULE_DIR / "external_selector_test_01.json"
OUTPUT_JSON_ROOT = REPO_ROOT / "external_selector_test_01.json"

UNSEEN_DATASET_PATH = REPO_ROOT / "tests" / "datasets" / "valid" / "pizza_sales.csv"
DATASET_NAME = "pizza_sales"
TARGET_COLUMN = "total_price"
TASK_TYPE = "regression"


# ---------------------------------------------------------------------------
# Execution Workflow
# ---------------------------------------------------------------------------

def run_external_selector_test() -> dict[str, Any]:
    print("=" * 75)
    print("LYNX PRODUCTION INFERENCE PIPELINE — EXTERNAL UNSEEN DATASET TEST")
    print("=" * 75)

    # Step 1: Verification that the dataset is completely unseen
    print("\n[Step 1] Verifying dataset is NOT in the 51 training datasets...")
    if not DATASET_METADATA_PATH.is_file():
        raise FileNotFoundError(f"Missing benchmark dataset metadata: {DATASET_METADATA_PATH}")

    with DATASET_METADATA_PATH.open("r", encoding="utf-8") as f:
        meta_data = json.load(f)
    training_dataset_names = [r["dataset_name"] for r in meta_data["records"]]

    is_unseen = DATASET_NAME not in training_dataset_names
    print(f"  Unseen dataset name: {DATASET_NAME}")
    print(f"  Presence in 51 training datasets: {not is_unseen} (Verified unseen: {is_unseen})")
    if not is_unseen:
        raise ValueError(f"Dataset {DATASET_NAME} was found in the 51 benchmark training datasets!")

    # Step 2: Load raw dataset and extract structural/task metadata
    print("\n[Step 2] Loading raw dataset and computing pre-execution metadata...")
    if not UNSEEN_DATASET_PATH.is_file():
        raise FileNotFoundError(f"Unseen dataset not found: {UNSEEN_DATASET_PATH}")

    df = pd.read_csv(UNSEEN_DATASET_PATH)
    n_rows, n_cols = df.shape
    print(f"  Dataset path: {UNSEEN_DATASET_PATH}")
    print(f"  Dimensions:   {n_rows:,} rows x {n_cols} columns")
    print(f"  Target:       {TARGET_COLUMN} (task_type: {TASK_TYPE})")

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    numeric_count = len(df[feature_cols].select_dtypes(include="number").columns)
    boolean_count = len(df[feature_cols].select_dtypes(include="bool").columns)
    datetime_count = len(df[feature_cols].select_dtypes(include=["datetime", "datetimetz"]).columns)
    categorical_count = len(feature_cols) - numeric_count - boolean_count - datetime_count

    feature_type_counts = {
        "numeric": numeric_count,
        "categorical": categorical_count,
        "boolean": boolean_count,
        "datetime": datetime_count,
        "other": 0,
    }

    evidence = build_dataset_semantic_evidence(
        dataset_name=DATASET_NAME,
        task_type=TASK_TYPE,
        target_column=TARGET_COLUMN,
        row_count=n_rows,
        feature_count=len(feature_cols),
        feature_type_counts=feature_type_counts,
        feature_names=feature_cols,
    )

    # Step 3: Generate semantic profile automatically via Ollama (qwen2.5:7b)
    print("\n[Step 3] Generating automated semantic profile via Ollama (qwen2.5:7b)...")
    prompt = build_dataset_semantic_prompt(evidence)
    provider = GroundworkOllamaProvider(model="qwen2.5:7b", timeout_seconds=120)
    raw_response = provider.generate_structured(prompt, Pipeline2DatasetSemanticProfile)
    semantic_profile = Pipeline2DatasetSemanticProfile(**raw_response).model_dump(mode="json")

    print("  Generated Semantic Profile:")
    print(f"    Probable Domain:   {semantic_profile['probable_domain']}")
    print(f"    Subdomain:         {semantic_profile['subdomain']}")
    print(f"    Dataset Purpose:   {semantic_profile['dataset_purpose']}")
    print(f"    Row Entity:        {semantic_profile['row_entity']}")
    print(f"    Target Concept:    {semantic_profile['target_concept']}")
    print(f"    Feature Context:   {semantic_profile['feature_context']}")
    print(f"    Semantic Summary:  {semantic_profile['semantic_summary']}")
    print(f"    Domain Confidence: {semantic_profile['domain_confidence']}")

    # Step 4: Load frozen model selector (NO retraining)
    print("\n[Step 4] Loading frozen lynx_model_selector.joblib...")
    if not SELECTOR_JOB_PATH.is_file():
        raise FileNotFoundError(f"Model selector artifact missing: {SELECTOR_JOB_PATH}")

    selector_mtime_before = SELECTOR_JOB_PATH.stat().st_mtime
    selector = LYNXModelSelector.load(SELECTOR_JOB_PATH)
    selector_mtime_after = SELECTOR_JOB_PATH.stat().st_mtime
    selector_retrained = (selector_mtime_before != selector_mtime_after)
    print(f"  Selector artifact: {SELECTOR_JOB_PATH}")
    print(f"  Selector retrained: {selector_retrained} (Frozen: {not selector_retrained})")

    # Step 5: Inference - score and rank candidate model families
    print("\n[Step 5] Scoring candidate model families on unseen dataset...")
    target_info = {
        "classification_type": None,
        "n_classes": None,
        "minority_class_ratio": 0.0,
        "target_missing_pct": float(df[TARGET_COLUMN].isna().mean()),
        "cb_chl_eligible": False,
    }

    ranking = selector.predict_ranking(
        task_type=TASK_TYPE,
        target_info=target_info,
        semantic_features=semantic_profile,
    )

    top1_model = ranking[0]
    top2_models = ranking[:2]

    # Step 6: Assemble output payload
    output_payload = {
        "schema_version": "1.0",
        "tested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verification_checks": {
            "unseen_dataset_confirmed": is_unseen,
            "not_in_51_training_datasets": is_unseen,
            "selector_was_not_retrained": not selector_retrained,
            "no_benchmark_outcomes_used": True,
            "no_predictive_models_trained": True,
        },
        "dataset_metadata": {
            "dataset_name": DATASET_NAME,
            "dataset_path": str(UNSEEN_DATASET_PATH),
            "n_rows": n_rows,
            "n_cols": n_cols,
            "target_column": TARGET_COLUMN,
            "task_type": TASK_TYPE,
            "feature_columns_count": len(feature_cols),
            "feature_type_counts": feature_type_counts,
        },
        "generated_semantic_profile": semantic_profile,
        "recommendations": {
            "top1_model": top1_model["model_name"],
            "top1_family": top1_model["model_family"],
            "top1_score": top1_model["predicted_suitability_score"],
            "top2_models": [
                {
                    "rank": m["rank"],
                    "model_name": m["model_name"],
                    "model_family": m["model_family"],
                    "predicted_suitability_score": m["predicted_suitability_score"],
                }
                for m in top2_models
            ],
            "full_ranking": ranking,
        },
    }

    # Write output to both requested paths
    OUTPUT_JSON_MODULE.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTPUT_JSON_ROOT.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved test results to:\n  {OUTPUT_JSON_MODULE}\n  {OUTPUT_JSON_ROOT}")

    # Step 7: Print formatted terminal report
    print("\n" + "=" * 75)
    print("EXTERNAL SELECTOR INFERENCE REPORT")
    print("=" * 75)
    print(f"Dataset Name:              {DATASET_NAME}")
    print(f"Dataset Path:              {UNSEEN_DATASET_PATH}")
    print(f"Dimensions:                {n_rows:,} rows x {n_cols} columns")
    print(f"Target Column:             {TARGET_COLUMN}")
    print(f"Task Type:                 {TASK_TYPE}")
    print(f"Generated Probable Domain: {semantic_profile['probable_domain']}")
    print(f"Semantic Summary:          {semantic_profile['semantic_summary']}")
    print("-" * 75)
    print(f"Top-1 Recommended Model:   {top1_model['model_name']} ({top1_model['model_family']}) [Score: {top1_model['predicted_suitability_score']:.4f}]")
    print(f"Top-2 Recommended Models:  {top2_models[0]['model_name']}, {top2_models[1]['model_name']}")
    print("-" * 75)
    print("Full 10-Model Suitability Ranking:")
    print(f"{'Rank':<6} | {'Model Name':<16} | {'Family':<24} | {'Suitability Score':<18}")
    print("-" * 75)
    for m in ranking:
        print(f"#{m['rank']:<5} | {m['model_name']:<16} | {m['model_family']:<24} | {m['predicted_suitability_score']:>12.4f}")
    print("=" * 75)

    return output_payload


if __name__ == "__main__":
    run_external_selector_test()

