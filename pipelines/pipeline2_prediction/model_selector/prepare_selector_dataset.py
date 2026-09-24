"""Prepare the meta-learning dataset for the LYNX model selector.

This script integrates:
  1. Structural/statistical dataset features (S) from dataset_metadata.json
  2. Semantic dataset features (D) from dataset_semantics.json
  3. Task and target metadata from dataset_target_registry.json
  4. Model capabilities and constraints from model_capabilities.json
  5. Empirical benchmark evaluations from model_performance.json

It outputs a clean, joined dataset-model level artifact at:
  pipelines/pipeline2_prediction/model_selector/selector_dataset.json
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent
PIPELINE2_DIR = MODULE_DIR.parent
INPUTS_DIR = PIPELINE2_DIR / "inputs"
TRAINING_DATA_DIR = PIPELINE2_DIR / "training_data"

METADATA_PATH = INPUTS_DIR / "dataset_metadata.json"
SEMANTICS_PATH = INPUTS_DIR / "dataset_semantics.json"
TARGET_REGISTRY_PATH = INPUTS_DIR / "dataset_target_registry.json"
CAPABILITIES_PATH = INPUTS_DIR / "model_capabilities.json"
PERFORMANCE_PATH = TRAINING_DATA_DIR / "model_performance.json"
OUTPUT_PATH = MODULE_DIR / "selector_dataset.json"


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON from a file with UTF-8 encoding."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    """Save JSON atomically to prevent partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Main Dataset Preparation
# ---------------------------------------------------------------------------

def prepare_selector_dataset() -> dict[str, Any]:
    """Load inputs, validate joins, build dataset-model records, and persist."""
    print("=" * 70)
    print("LYNX MODEL SELECTOR — DATASET PREPARATION")
    print("=" * 70)

    # 1. Load inputs
    print("\n[1/4] Loading input files...")
    meta_payload = _load_json(METADATA_PATH)
    sem_payload = _load_json(SEMANTICS_PATH)
    target_payload = _load_json(TARGET_REGISTRY_PATH)
    cap_payload = _load_json(CAPABILITIES_PATH)
    perf_payload = _load_json(PERFORMANCE_PATH)

    meta_records = meta_payload.get("records", [])
    sem_records = sem_payload.get("records", [])
    target_records = target_payload.get("datasets", [])
    cap_records = cap_payload.get("records", [])
    perf_records = perf_payload.get("records", [])

    # 2. Build lookup maps keyed by normalized string IDs
    meta_by_id = {str(r["dataset_id"]): r for r in meta_records}
    sem_by_id = {str(r["dataset_id"]): r for r in sem_records}
    target_by_id = {str(r["dataset_id"]): r for r in target_records}
    cap_by_id = {str(r["model_id"]): r for r in cap_records}

    datasets_found = len(target_by_id)
    models_found = len(cap_by_id)
    total_perf_records = len(perf_records)

    print(f"  - Datasets found in registry:   {datasets_found}")
    print(f"  - Datasets found in metadata:   {len(meta_by_id)}")
    print(f"  - Datasets found in semantics:  {len(sem_by_id)}")
    print(f"  - Models found in capabilities: {models_found}")
    print(f"  - Benchmark records found:      {total_perf_records}")

    # 3. Join validation
    print("\n[2/4] Validating join integrity...")
    missing_joins = 0
    duplicate_joins = 0
    seen_keys: set[tuple[str, str]] = set()

    for r in perf_records:
        did = str(r.get("dataset_id"))
        mid = str(r.get("model_id"))
        key = (did, mid)

        # Check duplicates
        if key in seen_keys:
            duplicate_joins += 1
        seen_keys.add(key)

        # Check missing joins across all 4 metadata registries
        if did not in target_by_id or did not in meta_by_id or did not in sem_by_id:
            missing_joins += 1
        if mid not in cap_by_id:
            missing_joins += 1

    # 4. Build records at dataset-model level
    print("\n[3/4] Assembling dataset-model records...")
    combined_records: list[dict[str, Any]] = []
    successful_records = 0
    failed_records = 0

    for perf_rec in perf_records:
        did = str(perf_rec["dataset_id"])
        mid = str(perf_rec["model_id"])

        meta_data = meta_by_id.get(did, {})
        sem_data = sem_by_id.get(did, {}).get("semantic_meta_features", {})
        target_data = target_by_id.get(did, {})
        cap_data = cap_by_id.get(mid, {})

        # Success determination: successful runs have metrics and no failure reason
        has_metrics = (
            perf_rec.get("classification_metrics") is not None
            or perf_rec.get("regression_metrics") is not None
        )
        is_success = has_metrics and (perf_rec.get("infeasible_reason") is None)

        if is_success:
            successful_records += 1
        else:
            failed_records += 1

        # Preserve structural features from dataset_metadata.json
        structural_features = {
            "n_rows": meta_data.get("n_rows"),
            "n_features": meta_data.get("n_features"),
            "numeric_feature_count": meta_data.get("numeric_feature_count"),
            "categorical_feature_count": meta_data.get("categorical_feature_count"),
            "boolean_feature_count": meta_data.get("boolean_feature_count"),
            "datetime_feature_count": meta_data.get("datetime_feature_count"),
            "other_feature_count": meta_data.get("other_feature_count"),
            "numeric_feature_ratio": meta_data.get("numeric_feature_ratio"),
            "categorical_feature_ratio": meta_data.get("categorical_feature_ratio"),
            "boolean_feature_ratio": meta_data.get("boolean_feature_ratio"),
            "datetime_feature_ratio": meta_data.get("datetime_feature_ratio"),
            "other_feature_ratio": meta_data.get("other_feature_ratio"),
            "overall_missing_pct": meta_data.get("overall_missing_pct"),
            "features_with_missing_count": meta_data.get("features_with_missing_count"),
            "features_with_missing_ratio": meta_data.get("features_with_missing_ratio"),
            "rows_with_missing_ratio": meta_data.get("rows_with_missing_ratio"),
            "mean_feature_cardinality": meta_data.get("mean_feature_cardinality"),
            "median_feature_cardinality": meta_data.get("median_feature_cardinality"),
            "mean_unique_ratio": meta_data.get("mean_unique_ratio"),
            "high_cardinality_feature_ratio": meta_data.get("high_cardinality_feature_ratio"),
        }

        # Preserve semantic features from dataset_semantics.json
        semantic_features = {
            "probable_domain": sem_data.get("probable_domain"),
            "subdomain": sem_data.get("subdomain"),
            "dataset_purpose": sem_data.get("dataset_purpose"),
            "row_entity": sem_data.get("row_entity"),
            "target_concept": sem_data.get("target_concept"),
            "feature_context": sem_data.get("feature_context"),
            "semantic_summary": sem_data.get("semantic_summary"),
            "domain_confidence": sem_data.get("domain_confidence"),
        }

        # Target and task metadata
        target_info = {
            "target_column": target_data.get("target_column"),
            "task_type": target_data.get("task_type") or perf_rec.get("task_type"),
            "classification_type": target_data.get("classification_type"),
            "n_classes": target_data.get("n_classes"),
            "class_distribution": target_data.get("class_distribution"),
            "minority_class_ratio": target_data.get("minority_class_ratio"),
            "target_row_count": target_data.get("target_row_count"),
            "target_missing_count": target_data.get("target_missing_count"),
            "target_missing_pct": target_data.get("target_missing_pct"),
            "imbalance_category": target_data.get("imbalance_category"),
            "cb_chl_eligible": target_data.get("cb_chl_eligible"),
        }

        # Model capability information
        model_capability_info = {
            "model_family": cap_data.get("model_family"),
            "supported_task_types": cap_data.get("supported_task_types"),
            "hard_limits": cap_data.get("hard_limits"),
            "capabilities": cap_data.get("capabilities"),
            "runtime_and_provisioning": cap_data.get("runtime_and_provisioning"),
            "empirical_guidance": cap_data.get("empirical_guidance"),
        }

        # Observed benchmark performance (metrics preserved as-is; failures kept as null)
        observed_performance = {
            "primary_metric_name": perf_rec.get("primary_metric_name"),
            "primary_metric_value": perf_rec.get("primary_metric_value"),
            "baseline_relative_score": perf_rec.get("baseline_relative_score"),
            "train_runtime_seconds": perf_rec.get("train_runtime_seconds"),
            "inference_runtime_seconds": perf_rec.get("inference_runtime_seconds"),
            "classification_metrics": perf_rec.get("classification_metrics"),
            "regression_metrics": perf_rec.get("regression_metrics"),
            "hyperparams": perf_rec.get("hyperparams"),
        }

        record = {
            "dataset_id": int(did) if did.isdigit() else did,
            "dataset_name": target_data.get("dataset_name") or meta_data.get("dataset_name"),
            "model_name": mid,
            "task_type": perf_rec.get("task_type") or target_data.get("task_type"),
            "target_info": target_info,
            "structural_features": structural_features,
            "semantic_features": semantic_features,
            "model_capabilities": model_capability_info,
            "benchmark_performance": observed_performance,
            "run_status": "success" if is_success else "failure",
            "success": is_success,
            "failure_reason": perf_rec.get("infeasible_reason") if not is_success else None,
        }
        combined_records.append(record)

    # 5. Persist output artifact
    print("\n[4/4] Saving output to selector_dataset.json...")
    output_payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": {
            "number_of_datasets_found": datasets_found,
            "number_of_models_found": models_found,
            "number_of_benchmark_records": total_perf_records,
            "successful_records": successful_records,
            "failed_records": failed_records,
            "missing_joins": missing_joins,
            "duplicate_joins": duplicate_joins,
        },
        "records": combined_records,
    }
    _atomic_save_json(OUTPUT_PATH, output_payload)

    # 6. Final Report
    print("\n" + "=" * 70)
    print("JOIN VALIDATION REPORT")
    print("=" * 70)
    print(f"  Number of datasets found:             {datasets_found}")
    print(f"  Number of models found:               {models_found}")
    print(f"  Total dataset-model benchmark records: {total_perf_records}")
    print(f"  Successful records:                   {successful_records}")
    print(f"  Failed records:                       {failed_records}")
    print(f"  Missing joins:                        {missing_joins}")
    print(f"  Duplicate joins:                      {duplicate_joins}")
    print(f"  Output file:                          {OUTPUT_PATH}")
    print("=" * 70)

    return output_payload


if __name__ == "__main__":
    prepare_selector_dataset()

