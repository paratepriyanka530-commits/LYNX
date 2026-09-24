"""Canonical TabArena benchmark execution and exact-schema checkpointing."""

from __future__ import annotations

import gc
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from benchmarks.tabarena.evaluation_metrics import (
    classification_results,
    regression_results,
)
from benchmarks.tabarena.pipeline2_groundwork import DATASET_DIR, feasibility


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "pipelines" / "pipeline2_prediction" / "inputs"
PERFORMANCE_PATH = (
    ROOT / "pipelines" / "pipeline2_prediction" / "training_data"
    / "model_performance.json"
)
TARGET_REGISTRY_PATH = INPUT_DIR / "dataset_target_registry.json"

CANONICAL_MODEL_IDS = (
    "random_forest", "xgboost", "lightgbm", "catboost", "xtab",
    "tabpfn", "tabm", "tabiclv2", "realmlp", "ft_transformer",
)
RECORD_FIELDS = (
    "dataset_id", "model_id", "task_type", "feasible",
    "infeasible_reason", "classification_metrics", "regression_metrics",
    "primary_metric_name", "primary_metric_value",
    "baseline_relative_score", "train_runtime_seconds",
    "inference_runtime_seconds", "split_seed", "source", "hyperparams",
)
CLASSIFICATION_FIELDS = (
    "accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
    "f1_macro", "f2_macro", "roc_auc", "log_loss", "mcc",
    "confusion_matrix",
)
REGRESSION_FIELDS = ("rmse", "mae", "r2", "nrmse")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "record_schema": {
            "fields": list(RECORD_FIELDS),
            "classification_metric_fields": list(CLASSIFICATION_FIELDS),
            "regression_metric_fields": list(REGRESSION_FIELDS),
            "runtime_failure_encoding": (
                "feasible=true, null metrics, and infeasible_reason prefixed "
                "with 'runtime_failure:' because the requested record schema "
                "has no separate status/failure field"
            ),
        },
        "records": records,
    }


def load_records(path: Path = PERFORMANCE_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError(f"Invalid benchmark checkpoint: {path}")
    for record in records:
        validate_record(record)
    keys = [(record["dataset_id"], record["model_id"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate dataset/model records in checkpoint: {path}")
    return records


def load_benchmark_datasets() -> list[dict[str, Any]]:
    payload = json.loads(TARGET_REGISTRY_PATH.read_text(encoding="utf-8"))
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 51:
        raise ValueError("Frozen target registry must contain exactly 51 datasets")
    result = []
    for item in datasets:
        if item.get("status") != "resolved":
            raise ValueError(f"Unresolved target registry row: {item.get('dataset_id')}")
        source = DATASET_DIR / item["dataset_file"]
        if not source.is_file():
            raise FileNotFoundError(f"Missing local TabArena CSV: {source}")
        result.append({
            "task_id": str(item["dataset_id"]),
            "dataset_name": item["dataset_name"],
            "csv_file": item["dataset_file"],
            "target_column": item["target_column"],
            "task_type": item["task_type"],
            "n_classes": item.get("n_classes"),
        })
    return result


def _fit_feature_preprocessor(model_id: str, X_train: pd.DataFrame):
    preprocessors = {
        "lightgbm": (
            "pipelines.pipeline2_prediction.models.lightgbm.lightgbm_preprocessor",
            "LightGBMPreprocessor",
        ),
        "xgboost": (
            "pipelines.pipeline2_prediction.models.xgboost.xgboost_preprocessor",
            "XGBoostPreprocessor",
        ),
        "catboost": (
            "pipelines.pipeline2_prediction.models.catboost.catboost_preprocessor",
            "CatBoostPreprocessor",
        ),
    }
    if model_id not in preprocessors:
        return X_train.copy(), None
    module_name, class_name = preprocessors[model_id]
    preprocessor = getattr(importlib.import_module(module_name), class_name)()
    return preprocessor.fit_transform(X_train), preprocessor


def _score(task_type: str, y_train: Any, y_test: Any, predictions: Any):
    if task_type == "classification":
        metric = float(balanced_accuracy_score(y_test, predictions))
        majority = pd.Series(y_train).mode().iloc[0]
        baseline_predictions = np.repeat(majority, len(y_test))
        baseline = float(balanced_accuracy_score(y_test, baseline_predictions))
        relative = (metric - baseline) / (1.0 - baseline) if baseline < 1.0 else 0.0
        return "balanced_accuracy", metric, float(relative)
    metric = float(mean_squared_error(y_test, predictions) ** 0.5)
    baseline_predictions = np.repeat(float(np.mean(y_train)), len(y_test))
    baseline = float(mean_squared_error(y_test, baseline_predictions) ** 0.5)
    relative = 1.0 - (metric / baseline) if baseline > 0 else 0.0
    return "rmse", metric, float(relative)


def _base_record(
    dataset: dict[str, Any], model_id: str, task_type: str, split_seed: int,
) -> dict[str, Any]:
    return {
        "dataset_id": str(dataset["task_id"]),
        "model_id": model_id,
        "task_type": task_type,
        "feasible": True,
        "infeasible_reason": None,
        "classification_metrics": None,
        "regression_metrics": None,
        "primary_metric_name": None,
        "primary_metric_value": None,
        "baseline_relative_score": None,
        "train_runtime_seconds": None,
        "inference_runtime_seconds": None,
        "split_seed": split_seed,
        "source": "team_original_run",
        "hyperparams": {},
    }


def _finite_or_none(value: Any) -> bool:
    return value is None or (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_record(record: dict[str, Any]) -> None:
    if tuple(record.keys()) != RECORD_FIELDS:
        raise ValueError(
            "Performance record fields differ from the requested schema: "
            f"{tuple(record.keys())}"
        )
    if record["model_id"] not in CANONICAL_MODEL_IDS:
        raise ValueError(f"Noncanonical model: {record['model_id']}")
    if record["task_type"] not in {"classification", "regression"}:
        raise ValueError(f"Invalid task type: {record['task_type']}")
    if record["source"] != "team_original_run":
        raise ValueError("Invalid benchmark source")
    if not isinstance(record["feasible"], bool):
        raise ValueError("feasible must be boolean")
    for field in (
        "primary_metric_value", "baseline_relative_score",
        "train_runtime_seconds", "inference_runtime_seconds",
    ):
        if not _finite_or_none(record[field]):
            raise ValueError(f"Non-finite {field} in {record['dataset_id']}/{record['model_id']}")
    classification = record["classification_metrics"]
    regression = record["regression_metrics"]
    if classification is not None:
        if tuple(classification.keys()) != CLASSIFICATION_FIELDS:
            raise ValueError("Classification metrics differ from requested schema")
        for key, value in classification.items():
            if key == "confusion_matrix":
                if not isinstance(value, list):
                    raise ValueError("confusion_matrix must be a list")
            elif not _finite_or_none(value):
                raise ValueError(f"Non-finite classification metric {key}")
    if regression is not None:
        if tuple(regression.keys()) != REGRESSION_FIELDS:
            raise ValueError("Regression metrics differ from requested schema")
        if any(not _finite_or_none(value) for value in regression.values()):
            raise ValueError("Non-finite regression metric")
    reason = record["infeasible_reason"]
    if record["feasible"] is False:
        if not reason or classification is not None or regression is not None:
            raise ValueError("Hard-infeasible record is malformed")
    elif reason is not None:
        if not str(reason).startswith("runtime_failure:"):
            raise ValueError("A feasible record reason must identify runtime_failure")
        if classification is not None or regression is not None:
            raise ValueError("Runtime failure must not contain fabricated metrics")
    elif record["task_type"] == "classification":
        if classification is None or regression is not None:
            raise ValueError("Successful classification record has wrong metric block")
        if record["primary_metric_name"] != "balanced_accuracy":
            raise ValueError("Classification primary metric must be balanced_accuracy")
    else:
        if regression is None or classification is not None:
            raise ValueError("Successful regression record has wrong metric block")
        if record["primary_metric_name"] != "rmse":
            raise ValueError("Regression primary metric must be rmse")


def _prepare_dataset(dataset: dict[str, Any], split_seed: int):
    source = DATASET_DIR / dataset["csv_file"]
    frame = pd.read_csv(source, low_memory=False)
    target = dataset["target_column"]
    if target not in frame.columns:
        raise ValueError(f"Exact target {target!r} is absent from {source.name}")
    frame = frame.dropna(subset=[target]).reset_index(drop=True)
    X = frame.drop(columns=[target])
    y = frame[target]
    task_type = dataset["task_type"]
    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=split_seed, stratify=stratify,
    )
    if task_type == "classification":
        target_encoder = LabelEncoder()
        y_train = target_encoder.fit_transform(y_train_raw.astype(str))
        y_test = target_encoder.transform(y_test_raw.astype(str))
        class_count = int(len(target_encoder.classes_))
    else:
        y_train = pd.to_numeric(y_train_raw, errors="raise").to_numpy()
        y_test = pd.to_numeric(y_test_raw, errors="raise").to_numpy()
        class_count = None
    return task_type, class_count, X_train, X_test, y_train, y_test


def _metrics_for_task(task_type: str, predictive: dict[str, Any]):
    if task_type == "classification":
        return {key: predictive.get(key) for key in CLASSIFICATION_FIELDS}, None
    return None, {key: predictive.get(key) for key in REGRESSION_FIELDS}


def _release_model_resources() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _dataset_records(records: list[dict[str, Any]], dataset_id: str):
    return [record for record in records if record["dataset_id"] == dataset_id]


def _finalize_relative_scores(
    records: list[dict[str, Any]], dataset_id: str,
    task_type: str, y_train: Any, y_test: Any,
) -> None:
    dataset_rows = _dataset_records(records, dataset_id)
    if len(dataset_rows) != len(CANONICAL_MODEL_IDS):
        return
    if task_type == "classification":
        majority = pd.Series(y_train).mode().iloc[0]
        baseline = float(balanced_accuracy_score(
            y_test, np.repeat(majority, len(y_test))
        ))
        for record in dataset_rows:
            value = record["primary_metric_value"]
            record["baseline_relative_score"] = None if value is None else float(
                (value - baseline) / (1.0 - baseline) if baseline < 1.0 else 0.0
            )
    else:
        baseline = float(mean_squared_error(
            y_test, np.repeat(float(np.mean(y_train)), len(y_test))
        ) ** 0.5)
        for record in dataset_rows:
            value = record["primary_metric_value"]
            record["baseline_relative_score"] = None if value is None else float(
                1.0 - (value / baseline) if baseline > 0 else 0.0
            )


def run_evaluations(
    datasets: list[dict[str, Any]],
    output_path: Path = PERFORMANCE_PATH,
    split_seed: int = 42,
    model_ids: tuple[str, ...] = CANONICAL_MODEL_IDS,
    retry_runtime_failures: bool = False,
) -> list[dict[str, Any]]:
    """Run sequentially and atomically persist one record per combination."""
    output_path = Path(output_path)
    records = load_records(output_path)
    requested_keys = {
        (str(dataset["task_id"]), model_id)
        for dataset in datasets for model_id in model_ids
    }
    if retry_runtime_failures:
        retained = [
            row for row in records
            if not (
                (row["dataset_id"], row["model_id"]) in requested_keys
                and row["feasible"] is True
                and str(row["infeasible_reason"] or "").startswith("runtime_failure:")
            )
        ]
        if len(retained) != len(records):
            records = retained
            _atomic_json(output_path, _artifact(records))
    completed = {(row["dataset_id"], row["model_id"]) for row in records}
    selected_keys = {
        (str(dataset["task_id"]), model_id)
        for dataset in datasets for model_id in model_ids
    }
    total = len(selected_keys)

    def persist(record: dict[str, Any], dataset_index: int, model_index: int) -> None:
        validate_record(record)
        key = (record["dataset_id"], record["model_id"])
        if key in completed:
            return
        records.append(record)
        completed.add(key)
        _atomic_json(output_path, _artifact(records))
        outcome = (
            "infeasible" if not record["feasible"] else
            "failure" if record["infeasible_reason"] else "success"
        )
        metric = (
            f" | {record['primary_metric_name']}={record['primary_metric_value']:.6g}"
            if record["primary_metric_value"] is not None else ""
        )
        runtime = (
            (record["train_runtime_seconds"] or 0)
            + (record["inference_runtime_seconds"] or 0)
        )
        print(
            f"Dataset {dataset_index}/{len(datasets)} | Model {model_index}/10 | "
            f"{record['model_id']} | {outcome}{metric} | {runtime:.1f}s",
            flush=True,
        )
        selected = [
            row for row in records
            if (row["dataset_id"], row["model_id"]) in selected_keys
        ]
        successes = sum(row["feasible"] and row["infeasible_reason"] is None for row in selected)
        infeasible = sum(not row["feasible"] for row in selected)
        failures = sum(row["feasible"] and row["infeasible_reason"] is not None for row in selected)
        print(
            f"Completed {len(selected)}/{total} | remaining={total-len(selected)} | "
            f"success={successes} | infeasible={infeasible} | failure={failures}",
            flush=True,
        )

    for dataset_index, dataset in enumerate(datasets, 1):
        dataset_id = str(dataset["task_id"])
        pending = [
            model_id for model_id in model_ids
            if (dataset_id, model_id) not in completed
        ]
        needs_relative_finalize = any(
            row["baseline_relative_score"] is None
            and row["primary_metric_value"] is not None
            for row in _dataset_records(records, dataset_id)
        )
        if not pending and not needs_relative_finalize:
            print(f"[SKIP] Dataset {dataset_index}/{len(datasets)} task_{dataset_id} complete", flush=True)
            continue
        try:
            task_type, class_count, X_train, X_test, y_train, y_test = _prepare_dataset(
                dataset, split_seed
            )
        except Exception as exc:
            for model_index, model_id in enumerate(model_ids, 1):
                if model_id not in pending:
                    continue
                record = _base_record(dataset, model_id, dataset["task_type"], split_seed)
                record["infeasible_reason"] = (
                    f"runtime_failure:dataset_setup:{type(exc).__name__}:{exc}"
                )
                persist(record, dataset_index, model_index)
            continue

        for model_index, model_id in enumerate(model_ids, 1):
            if model_id not in pending:
                print(
                    f"Dataset {dataset_index}/{len(datasets)} | Model {model_index}/10 | "
                    f"{model_id} | resume_skip",
                    flush=True,
                )
                continue
            record = _base_record(dataset, model_id, task_type, split_seed)
            allowed, reason = feasibility(
                model_id, len(X_train), X_train.shape[1], task_type, class_count
            )
            if allowed is not True:
                if allowed is False:
                    record["feasible"] = False
                    record["infeasible_reason"] = str(reason)
                else:
                    record["infeasible_reason"] = f"runtime_failure:feasibility:{reason}"
                persist(record, dataset_index, model_index)
                continue

            model = None
            preprocessor = None
            try:
                package = importlib.import_module(
                    f"pipelines.pipeline2_prediction.models.{model_id}"
                )
                train_started = time.perf_counter()
                X_train_model, preprocessor = _fit_feature_preprocessor(model_id, X_train)
                overrides: dict[str, Any] = {}
                model = package.build_model(task_type, split_seed, **overrides)
                package.train_model(model, X_train_model, y_train)
                record["train_runtime_seconds"] = float(time.perf_counter() - train_started)

                inference_started = time.perf_counter()
                X_test_model = (
                    preprocessor.transform(X_test)
                    if preprocessor is not None else X_test.copy()
                )
                predictions = np.asarray(
                    package.predict_model(model, X_test_model)
                ).reshape(-1)
                if task_type == "classification":
                    predictions = pd.to_numeric(
                        pd.Series(predictions), errors="raise"
                    ).to_numpy(dtype=np.asarray(y_test).dtype)
                    predictive, _calibration, _probability = classification_results(
                        y_train, y_test, predictions, model_id, model, X_test_model
                    )
                else:
                    predictive, _calibration, _probability = regression_results(
                        y_test, predictions
                    )
                record["inference_runtime_seconds"] = float(
                    time.perf_counter() - inference_started
                )
                classification_metrics, regression_metrics = _metrics_for_task(
                    task_type, predictive
                )
                primary_name, primary_value, _relative = _score(
                    task_type, y_train, y_test, predictions
                )
                record.update({
                    "classification_metrics": classification_metrics,
                    "regression_metrics": regression_metrics,
                    "primary_metric_name": primary_name,
                    "primary_metric_value": primary_value,
                    "hyperparams": overrides,
                })
            except Exception as exc:
                record.update({
                    "classification_metrics": None,
                    "regression_metrics": None,
                    "primary_metric_name": None,
                    "primary_metric_value": None,
                    "baseline_relative_score": None,
                    "infeasible_reason": (
                        f"runtime_failure:train_or_evaluate:{type(exc).__name__}:{exc}"
                    ),
                })
            persist(record, dataset_index, model_index)
            del model, preprocessor
            _release_model_resources()

        _finalize_relative_scores(records, dataset_id, task_type, y_train, y_test)
        for row in _dataset_records(records, dataset_id):
            validate_record(row)
        _atomic_json(output_path, _artifact(records))
    return records


def validate_complete_matrix(records: list[dict[str, Any]]) -> dict[str, Any]:
    datasets = load_benchmark_datasets()
    expected = {
        (str(dataset["task_id"]), model_id)
        for dataset in datasets for model_id in CANONICAL_MODEL_IDS
    }
    keys = {(row["dataset_id"], row["model_id"]) for row in records}
    if len(records) != 510 or keys != expected:
        raise ValueError(
            f"Expected exact 510-pair matrix; rows={len(records)}, "
            f"missing={len(expected - keys)}, extra={len(keys - expected)}"
        )
    task_by_id = {str(row["task_id"]): row["task_type"] for row in datasets}
    for record in records:
        validate_record(record)
        if record["task_type"] != task_by_id[record["dataset_id"]]:
            raise ValueError(f"Wrong task type for dataset {record['dataset_id']}")
    return {
        "total": 510,
        "success": sum(r["feasible"] and r["infeasible_reason"] is None for r in records),
        "infeasible": sum(not r["feasible"] for r in records),
        "failure": sum(r["feasible"] and r["infeasible_reason"] is not None for r in records),
    }


__all__ = [
    "CANONICAL_MODEL_IDS", "CLASSIFICATION_FIELDS", "PERFORMANCE_PATH",
    "RECORD_FIELDS", "REGRESSION_FIELDS", "load_benchmark_datasets",
    "load_records", "run_evaluations", "validate_complete_matrix",
    "validate_record",
]
