"""Build and validate authoritative, leakage-safe selector input artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tabarena.pipeline2_groundwork import (  # noqa: E402
    DATASET_DIR,
    EVALUATION_MODEL_IDS,
    MODEL_CAPABILITIES,
    OUTPUT_DIR,
)
INPUT_DIR = ROOT / "pipelines" / "pipeline2_prediction" / "inputs"
TRAINING_DATA_DIR = ROOT / "pipelines" / "pipeline2_prediction" / "training_data"
TARGET_PATH = INPUT_DIR / "dataset_target_registry.json"
MANIFEST_PATH = OUTPUT_DIR / "dataset_manifest.json"
STATISTICS_PATH = OUTPUT_DIR / "statistical_meta_features.json"
DATASET_SEMANTIC_PROFILE_DIR = OUTPUT_DIR / "dataset_semantic_profiles"
INVENTORY_PATH = ROOT / "benchmarks" / "tabarena" / "model_inventory.json"

MODEL_SELECTOR_FACTS = {
    "random_forest": {"family": "tree_ensemble", "missing": None, "categorical": False, "oob": True, "gpu_optional": False},
    "xgboost": {"family": "gradient_boosted_trees", "missing": False, "categorical": False, "oob": False, "gpu_optional": False},
    "lightgbm": {"family": "gradient_boosted_trees", "missing": False, "categorical": False, "oob": False, "gpu_optional": False},
    "catboost": {"family": "gradient_boosted_trees", "missing": False, "categorical": True, "oob": False, "gpu_optional": False},
    "xtab": {"family": "pretrained_transformer", "missing": False, "categorical": True, "oob": False, "gpu_optional": True},
    "tabpfn": {"family": "prior_fitted_network", "missing": True, "categorical": None, "oob": False, "gpu_optional": True},
    "tabm": {"family": "deep_tabular_ensemble", "missing": False, "categorical": False, "oob": False, "gpu_optional": True},
    "tabiclv2": {"family": "prior_fitted_network", "missing": True, "categorical": True, "oob": False, "gpu_optional": True},
    "realmlp": {"family": "multilayer_perceptron", "missing": False, "categorical": True, "oob": False, "gpu_optional": True},
    "ft_transformer": {"family": "transformer", "missing": False, "categorical": True, "oob": False, "gpu_optional": True},
}

PROVISIONED_MODELS = {"xtab", "tabpfn", "tabiclv2"}
PRETRAINED_MODELS = {"xtab", "tabpfn", "tabiclv2"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_records() -> list[dict[str, Any]]:
    payload = _read(TARGET_PATH)
    records = payload.get("datasets", [])
    if len(records) != 51 or payload.get("validation", {}).get("passed") is not True:
        raise ValueError("The authoritative Phase C target registry is not a validated 51-row artifact")
    return records


def build_dataset_metadata() -> dict[str, Any]:
    targets = _target_records()
    manifest = _read(MANIFEST_PATH)["datasets"]
    by_id = {int(item["task_id"]): item for item in manifest}
    records = []
    for target in targets:
        dataset_id = int(target["dataset_id"])
        source = DATASET_DIR / target["dataset_file"]
        manifest_row = by_id.get(dataset_id)
        if manifest_row is None or not source.is_file():
            raise ValueError(f"Dataset {dataset_id} cannot be resolved to its local CSV")
        if manifest_row["csv_file"] != target["dataset_file"]:
            raise ValueError(f"Dataset {dataset_id} filename disagrees between manifest and target registry")
        frame = pd.read_csv(source, low_memory=False)
        target_column = target["benchmark_target_column"]
        if target_column not in frame.columns:
            raise ValueError(f"Dataset {dataset_id} lacks exact benchmark target {target_column!r}")
        features = frame.drop(columns=[target_column])
        numeric_columns = [
            column for column in features
            if pd.api.types.is_numeric_dtype(features[column])
            and not pd.api.types.is_bool_dtype(features[column])
        ]
        boolean_columns = [column for column in features if pd.api.types.is_bool_dtype(features[column])]
        datetime_columns = [column for column in features if pd.api.types.is_datetime64_any_dtype(features[column])]
        categorical_columns = [
            column for column in features
            if column not in boolean_columns
            and column not in datetime_columns
            and (
                isinstance(features[column].dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(features[column])
                or pd.api.types.is_string_dtype(features[column])
            )
        ]
        classified = set(numeric_columns + boolean_columns + datetime_columns + categorical_columns)
        other_columns = [column for column in features if column not in classified]
        cardinality = features.nunique(dropna=True).astype(float)
        unique_ratios = cardinality / max(1, len(features))
        missing_by_feature = features.isna().any(axis=0)
        denominator = max(1, features.shape[1])
        records.append({
            "dataset_id": dataset_id,
            "dataset_name": target["dataset_name"],
            "dataset_file": target["dataset_file"],
            "openml_dataset_id": int(manifest_row["openml_dataset_id"]),
            "n_rows": int(len(frame)),
            "n_features": int(features.shape[1]),
            "numeric_feature_count": len(numeric_columns),
            "categorical_feature_count": len(categorical_columns),
            "boolean_feature_count": len(boolean_columns),
            "datetime_feature_count": len(datetime_columns),
            "other_feature_count": len(other_columns),
            "numeric_feature_ratio": len(numeric_columns) / denominator,
            "categorical_feature_ratio": len(categorical_columns) / denominator,
            "boolean_feature_ratio": len(boolean_columns) / denominator,
            "datetime_feature_ratio": len(datetime_columns) / denominator,
            "other_feature_ratio": len(other_columns) / denominator,
            "overall_missing_pct": float(features.isna().to_numpy().mean() * 100) if features.size else 0.0,
            "features_with_missing_count": int(missing_by_feature.sum()),
            "features_with_missing_ratio": float(missing_by_feature.mean()) if len(missing_by_feature) else 0.0,
            "rows_with_missing_ratio": float(features.isna().any(axis=1).mean()) if len(features) else 0.0,
            "mean_feature_cardinality": float(cardinality.mean()) if len(cardinality) else 0.0,
            "median_feature_cardinality": float(cardinality.median()) if len(cardinality) else 0.0,
            "mean_unique_ratio": float(unique_ratios.mean()) if len(unique_ratios) else 0.0,
            "high_cardinality_feature_ratio": float((unique_ratios >= 0.9).mean()) if len(unique_ratios) else 0.0,
            "source_csv_sha256": _sha256(source),
        })
    return {
        "schema_version": "1.0",
        "generated_at": _now(),
        "dataset_count": len(records),
        "provenance": {
            "dataset_identity": "OpenML TabArena suite 457 manifest",
            "target_exclusion": "validated benchmark_target_column from Phase C registry",
            "statistics": "deterministically recomputed from local CSV predictor columns",
            "csv_dtype_policy": "pandas read_csv inference; no semantic date parsing",
        },
        "selector_availability": "INFERENCE_AVAILABLE",
        "records": records,
    }


def build_dataset_semantics() -> dict[str, Any]:
    """Build selector semantics exclusively from one-call Pipeline 2 profiles."""
    targets = _target_records()
    records = []
    for target in targets:
        dataset_id = int(target["dataset_id"])
        path = DATASET_SEMANTIC_PROFILE_DIR / f"task_{dataset_id}.json"
        if not path.is_file():
            records.append({
                "dataset_id": dataset_id,
                "dataset_name": target["dataset_name"],
                "status": "pending",
                "model": None,
                "semantic_meta_features": None,
                "semantic_evidence": None,
            })
            continue
        payload = _read(path)
        if payload.get("status") != "complete":
            records.append({
                "dataset_id": dataset_id,
                "dataset_name": target["dataset_name"],
                "status": payload.get("status", "pending"),
                "model": payload.get("model"),
                "semantic_meta_features": None,
                "semantic_evidence": payload.get("evidence"),
            })
            continue
        if payload.get("model") != "qwen2.5:7b":
            raise ValueError(
                f"Semantic profile for dataset {dataset_id} used {payload.get('model')!r}; "
                "the frozen selector input requires qwen2.5:7b"
            )
        if int(payload["dataset_id"]) != dataset_id or payload["dataset_name"] != target["dataset_name"]:
            raise ValueError(f"Semantic profile identity mismatch for dataset {dataset_id}")
        profile = payload["semantic_profile"]
        required = {
            "probable_domain", "subdomain", "dataset_purpose", "row_entity",
            "target_concept", "feature_context", "semantic_summary", "domain_confidence",
        }
        if set(profile) != required:
            raise ValueError(f"Dataset semantic schema mismatch for dataset {dataset_id}")
        records.append({
            "dataset_id": dataset_id,
            "dataset_name": target["dataset_name"],
            "status": "complete",
            "model": payload.get("model"),
            "semantic_meta_features": profile,
            "semantic_evidence": payload.get("evidence"),
        })
    complete = sum(record["status"] == "complete" for record in records)
    return {
        "schema_version": "2.0",
        "generated_at": _now(),
        "dataset_count": len(records),
        "complete_count": complete,
        "pending_count": len(records) - complete,
        "required_model": "qwen2.5:7b",
        "semantic_unit": "one holistic dataset-level profile; no per-column semantics",
        "profile_source": "pipeline2_artifacts/dataset_semantic_profiles",
        "selector_availability": "INFERENCE_AVAILABLE",
        "records": records,
    }


def build_model_capabilities() -> dict[str, Any]:
    inventory = _read(INVENTORY_PATH)
    inventory_by_id = {item["model_id"]: item for item in inventory["candidates"]}
    records = []
    for model_id in EVALUATION_MODEL_IDS:
        key = model_id.replace("_", "")
        capability = MODEL_CAPABILITIES[key]
        candidate = inventory_by_id[model_id]
        facts = MODEL_SELECTOR_FACTS[model_id]
        records.append({
            "model_id": model_id,
            "model_family": facts["family"],
            "supported_task_types": capability["task_type_supported"],
            "hard_limits": {
                "min_train_rows": capability["minimum_training_rows"],
                "max_train_rows": capability["maximum_training_rows"],
                "max_features": capability["maximum_features"],
                "max_classes": capability["maximum_classes"],
                "evidence_class": {
                    "min_train_rows": candidate["constraint_classification"]["minimum_rows"],
                    "max_train_rows": candidate["constraint_classification"]["maximum_rows"],
                    "max_features": candidate["constraint_classification"]["maximum_features"],
                    "max_classes": candidate["constraint_classification"]["maximum_classes"],
                },
            },
            "capabilities": {
                "handles_missing_natively": facts["missing"],
                "handles_categorical_natively": facts["categorical"],
                "supports_probabilities": capability["supports_classification_probabilities"],
                "probability_class_order_available": capability["probability_class_order_available"],
                "supports_oob": facts["oob"],
                "null_means": "NONE_VERIFIED",
            },
            "runtime_and_provisioning": {
                "pretrained": model_id in PRETRAINED_MODELS,
                "gpu_required": False,
                "gpu_optional": facts["gpu_optional"],
                "provisioning_required": model_id in PROVISIONED_MODELS,
                "provisioning": capability["provisioning"],
                "evidence_class": "PROVISIONING" if model_id in PROVISIONED_MODELS else "NONE_VERIFIED",
            },
            "empirical_guidance": {
                "value": capability["empirical_guidance"],
                "evidence_class": "EMPIRICAL",
                "used_as_hard_gate": False,
            },
            "evidence": {
                "capability_source": capability["source"],
                "inventory_implementation_source": candidate["implementation_source"],
                "lifecycle_test_status": candidate["lifecycle_test_status"],
            },
        })
    return {
        "schema_version": "1.0",
        "generated_at": _now(),
        "canonical_model_count": len(records),
        "canonical_model_ids": list(EVALUATION_MODEL_IDS),
        "selector_availability": "INFERENCE_AVAILABLE",
        "constraint_classes": inventory["constraint_class_legend"],
        "experimental_execution_strategies": inventory.get("experimental_execution_strategies", []),
        "noncanonical_modes": {
            "cb_chl": "LightGBM training mode; not a candidate model",
            "crumb": "experimental execution strategy attached to a base model",
        },
        "records": records,
    }


def performance_template() -> dict[str, Any]:
    nullable_number = {"type": ["number", "null"]}
    return {
        "schema_version": "1.0",
        "description": "Benchmark supervision only; an empty template until real runs complete.",
        "record_unit": "dataset x model x execution_strategy x run",
        "canonical_model_ids": list(EVALUATION_MODEL_IDS),
        "record_schema": {
            "type": "object",
            "required": [
                "dataset_id", "model_id", "base_model_id", "execution_strategy",
                "run_id", "task_type", "feasible", "status", "split",
                "infeasible_reason", "hyperparameters",
                "predictive_metrics", "calibration_metrics", "probability_metadata",
                "primary_metric", "runtime", "resource_usage",
                "model_specific_metrics", "protocol_provenance", "failure",
            ],
            "properties": {
                "dataset_id": {"type": "integer"},
                "model_id": {"type": "string"},
                "base_model_id": {"enum": list(EVALUATION_MODEL_IDS)},
                "execution_strategy": {"enum": ["standard", "crumb"]},
                "run_id": {"type": "string", "minLength": 1},
                "task_type": {"enum": ["classification", "regression"]},
                "feasible": {"type": "boolean"},
                "infeasible_reason": {"type": ["string", "null"]},
                "status": {"enum": ["success", "infeasible_skip", "failure"]},
                "split": {"type": "object", "required": ["train_rows", "test_rows", "test_size", "split_seed", "stratified"]},
                "predictive_metrics": {"type": ["object", "null"]},
                "calibration_metrics": {"type": ["object", "null"]},
                "probability_metadata": {"type": ["object", "null"]},
                "primary_metric": {
                    "type": "object", "required": ["name", "value", "direction"],
                    "properties": {"name": {"type": ["string", "null"]}, "value": nullable_number, "direction": {"enum": ["maximize", "minimize", None]}},
                },
                "runtime": {
                    "type": "object", "required": ["train_seconds", "inference_seconds"],
                    "properties": {"train_seconds": nullable_number, "inference_seconds": nullable_number},
                },
                "resource_usage": {"type": ["object", "null"]},
                "model_specific_metrics": {"type": "object", "properties": {"random_forest_oob_score": nullable_number}},
                "hyperparameters": {"type": "object"},
                "protocol_provenance": {"type": "object"},
                "failure": {"type": ["object", "null"]},
            },
        },
        "outcome_rules": {
            "infeasible_skip": "feasible=false; performance values remain null; never encode as zero",
            "failure": "failure object records stage, exception type, and message; performance values remain null",
            "success": "primary and applicable predictive metrics contain observed benchmark outcomes",
        },
        "selector_availability": "TRAINING_OUTCOME_ONLY",
        "records": [],
    }


def join_contract() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "dataset_join_key": "dataset_id",
        "model_join_key": "base_model_id",
        "run_identity": ["dataset_id", "model_id", "execution_strategy", "run_id"],
        "dataset_sources": ["dataset_metadata.json", "dataset_semantics.json", "dataset_target_registry.json"],
        "model_source": "model_capabilities.json",
        "supervision_source": "../training_data/model_performance.json",
        "canonical_dataset_count": 51,
        "canonical_model_count": 10,
        "potential_base_grid_size": 510,
        "filename_is_not_a_join_key": True,
        "infeasible_policy": "retain explicitly as infeasible_skip; never convert to zero performance",
    }


def leakage_audit() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "result": "PASS",
        "policy": "Only INFERENCE_AVAILABLE fields may enter the selector feature vector.",
        "INFERENCE_AVAILABLE": {
            "dataset_metadata": [
                "n_rows", "n_features", "numeric_feature_count", "categorical_feature_count",
                "boolean_feature_count", "datetime_feature_count", "other_feature_count",
                "numeric_feature_ratio", "categorical_feature_ratio", "boolean_feature_ratio",
                "datetime_feature_ratio", "other_feature_ratio", "overall_missing_pct",
                "features_with_missing_count", "features_with_missing_ratio", "rows_with_missing_ratio",
                "mean_feature_cardinality", "median_feature_cardinality", "mean_unique_ratio",
                "high_cardinality_feature_ratio",
            ],
            "dataset_semantics": [
                "probable_domain", "subdomain", "dataset_purpose", "row_entity",
                "target_concept", "feature_context", "semantic_summary",
                "domain_confidence",
            ],
            "dataset_target_registry": [
                "task_type", "classification_type", "n_classes", "class_distribution",
                "minority_class_ratio", "target_missing_count", "target_missing_pct",
                "imbalance_category", "cb_chl_eligible",
            ],
            "model_capabilities": [
                "model_family", "supported_task_types", "hard_limits", "capabilities",
                "runtime_and_provisioning", "empirical_guidance",
            ],
            "rationale": "Available from the supplied labeled training dataset and registered model before benchmark execution.",
        },
        "EXCLUDED_RAW_EVIDENCE": [
            "representative feature names", "dataset-level prompt evidence",
            "old per-column semantic profiles and aggregates",
        ],
        "JOIN_OR_PROVENANCE_ONLY": [
            "dataset_id", "dataset_name", "dataset_file", "openml_dataset_id",
            "model_id", "source_csv_sha256", "evidence", "provenance",
        ],
        "TRAINING_OUTCOME_ONLY": [
            "held-out accuracy", "held-out balanced_accuracy", "held-out RMSE",
            "Brier score", "ECE", "ROC-AUC", "log-loss", "MCC", "confusion matrix",
            "benchmark train/inference runtime", "resource usage", "OOB outcome",
            "baseline-relative score", "oracle rank", "run status", "failure information",
        ],
        "validation": {
            "benchmark_outcome_fields_present_in_dataset_inputs": False,
            "performance_template_used_as_inference_input": False,
        },
    }


def validate_outputs(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    targets = _target_records()
    target_ids = [int(record["dataset_id"]) for record in targets]
    metadata_ids = [record["dataset_id"] for record in payloads["metadata"]["records"]]
    semantic_ids = [record["dataset_id"] for record in payloads["semantics"]["records"]]
    model_ids = [record["model_id"] for record in payloads["capabilities"]["records"]]
    if len(target_ids) != 51 or len(set(target_ids)) != 51:
        raise ValueError("Target registry dataset IDs are not exactly 51 unique values")
    if metadata_ids != target_ids or semantic_ids != target_ids:
        raise ValueError("Dataset input files do not share the exact ordered dataset_id join")
    for semantic in payloads["semantics"]["records"]:
        if semantic["status"] == "complete":
            fields = set(semantic["semantic_meta_features"])
            expected = {
                "probable_domain", "subdomain", "dataset_purpose", "row_entity",
                "target_concept", "feature_context", "semantic_summary", "domain_confidence",
            }
            if fields != expected:
                raise ValueError(
                    f"Dataset semantic fields mismatch for dataset {semantic['dataset_id']}"
                )
    if model_ids != list(EVALUATION_MODEL_IDS) or len(set(model_ids)) != 10:
        raise ValueError("Model capabilities do not contain exactly the canonical 10-model roster")
    task_counts = {
        "classification": sum(record["task_type"] == "classification" for record in targets),
        "regression": sum(record["task_type"] == "regression" for record in targets),
        "binary": sum(record["classification_type"] == "binary" for record in targets),
        "multiclass": sum(record["classification_type"] == "multiclass" for record in targets),
    }
    if task_counts != {"classification": 38, "regression": 13, "binary": 30, "multiclass": 8}:
        raise ValueError(f"Target task counts are wrong: {task_counts}")
    discrepancies = {record["dataset_name"]: record for record in targets}
    expected = {
        "miami_housing": ("structure_quality", "SALE_PRC"),
        "physiochemical_protein": ("SpatialDistNK", "ResidualSize"),
    }
    for name, (provided, benchmark) in expected.items():
        row = discrepancies[name]
        if row["provided_target_column"] != provided or row["benchmark_target_column"] != benchmark:
            raise ValueError(f"Authoritative target discrepancy changed for {name}")
    forbidden = {
        "accuracy", "balanced_accuracy", "rmse", "brier_score", "ece",
        "runtime_seconds", "baseline_relative_score", "oracle_rank",
    }
    for source_name in ("metadata", "semantics", "capabilities"):
        keys: set[str] = set()
        stack: list[Any] = [payloads[source_name]]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                keys.update(value)
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        overlap = forbidden & keys
        if overlap:
            raise ValueError(f"Training outcomes leaked into {source_name}: {sorted(overlap)}")
    return {
        "passed": True,
        "dataset_count": 51,
        "semantic_complete_count": payloads["semantics"]["complete_count"],
        "model_count": 10,
        "base_grid_size": 510,
        "task_counts": task_counts,
    }


def performance_template() -> dict[str, Any]:
    """Return the exact canonical benchmark checkpoint schema."""
    from benchmarks.tabarena.tabarena_benchmark import (
        CLASSIFICATION_FIELDS,
        RECORD_FIELDS,
        REGRESSION_FIELDS,
    )
    return {
        "schema_version": "2.0",
        "description": "Observed benchmark supervision only; never an inference-time selector feature.",
        "record_schema": {
            "fields": list(RECORD_FIELDS),
            "classification_metric_fields": list(CLASSIFICATION_FIELDS),
            "regression_metric_fields": list(REGRESSION_FIELDS),
        },
        "selector_availability": "TRAINING_OUTCOME_ONLY",
        "records": [],
    }


def validate_performance_record(record: dict[str, Any]) -> None:
    """Validate one future benchmark record without manufacturing any outcome."""
    from benchmarks.tabarena.tabarena_benchmark import validate_record
    validate_record(record)
    return
    required = set(performance_template()["record_schema"]["required"])
    missing = required - set(record)
    if missing:
        raise ValueError(f"Performance record is missing required fields: {sorted(missing)}")
    if record["base_model_id"] not in EVALUATION_MODEL_IDS:
        raise ValueError("base_model_id is not one of the canonical 10 models")
    if record["task_type"] not in {"classification", "regression"}:
        raise ValueError(f"Unknown task type: {record['task_type']!r}")
    strategy = record["execution_strategy"]
    if strategy == "standard" and record["model_id"] != record["base_model_id"]:
        raise ValueError("Standard execution requires model_id == base_model_id")
    if strategy == "crumb":
        expected = {"tabpfn": "tabpfn_crumb", "tabiclv2": "tabiclv2_crumb"}
        if expected.get(record["base_model_id"]) != record["model_id"]:
            raise ValueError("CRUMB is only a strategy for TabPFN or TabICLv2")
    elif strategy != "standard":
        raise ValueError(f"Unknown execution strategy: {strategy!r}")
    status = record["status"]
    if status not in {"success", "infeasible_skip", "failure"}:
        raise ValueError(f"Unknown performance status: {status!r}")
    if status == "infeasible_skip" and record["feasible"] is not False:
        raise ValueError("infeasible_skip requires feasible=false")
    if status == "success" and record["feasible"] is not True:
        raise ValueError("success requires feasible=true")
    if status in {"infeasible_skip", "failure"} and record["primary_metric"]["value"] is not None:
        raise ValueError("Skipped or failed runs must retain null performance")
    if status == "failure" and not record["failure"]:
        raise ValueError("failure status requires structured failure information")
    if status != "failure" and record["failure"] is not None:
        raise ValueError("Only failure rows may contain failure information")


def build_all() -> dict[str, Any]:
    payloads = {
        "metadata": build_dataset_metadata(),
        "semantics": build_dataset_semantics(),
        "capabilities": build_model_capabilities(),
        "performance": performance_template(),
    }
    validation = validate_outputs(payloads)
    _write_json(INPUT_DIR / "dataset_metadata.json", payloads["metadata"])
    _write_json(INPUT_DIR / "dataset_semantics.json", payloads["semantics"])
    _write_json(INPUT_DIR / "model_capabilities.json", payloads["capabilities"])
    _write_json(INPUT_DIR / "selector_join_contract.json", join_contract())
    _write_json(INPUT_DIR / "selector_leakage_audit.json", leakage_audit())
    _write_json(TRAINING_DATA_DIR / "model_performance.json", payloads["performance"])
    _write_json(INPUT_DIR / "selector_input_validation.json", {
        "schema_version": "1.0", "generated_at": _now(), **validation,
    })
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(build_all(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
