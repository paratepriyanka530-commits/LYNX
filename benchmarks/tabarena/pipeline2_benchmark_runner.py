"""Run the canonical 51 x 10 TabArena benchmark with atomic resume support.

The CLI requires explicit dataset selectors unless ``--all`` is supplied.  This
prevents an accidental full TabArena sweep while still providing the full loop.
"""

from __future__ import annotations

import argparse
import csv
import gc
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from benchmarks.tabarena.evaluation_metrics import (
    classification_results,
    regression_results,
)

from benchmarks.tabarena.pipeline2_groundwork import (
    EVALUATION_MODEL_IDS,
    EXPERIMENTAL_EXECUTION_STRATEGIES,
    LOG_COLUMNS,
    OUTPUT_DIR,
    feasibility,
    crumb_feasibility,
    initialize_log,
    load_manifest,
    statistical_meta_features,
    validate_log_record,
)
from pipelines.pipeline2_prediction.models.artifact_io import training_artifact_dir


class Pipeline1PrivacyClearanceError(RuntimeError):
    """The dataset has no valid, privacy-cleared Pipeline 1 artifact."""


def load_privacy_cleared_dataset(
    dataset: dict[str, Any],
    pipeline1_output_root: Path,
) -> pd.DataFrame:
    """Load only a hash-verified Pipeline 1 output; never read raw input."""
    dataset_id = str(dataset["task_id"])
    artifact_dir = Path(pipeline1_output_root) / dataset_id
    report_path = artifact_dir / "cleaning_report.json"
    cleaned_path = artifact_dir / "cleaned_dataset.csv"
    if not report_path.is_file() or not cleaned_path.is_file():
        raise Pipeline1PrivacyClearanceError(
            f"Dataset {dataset_id!r} is missing Pipeline 1 artifacts under "
            f"{artifact_dir}. Raw-input fallback is prohibited."
        )

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Pipeline1PrivacyClearanceError(
            f"Dataset {dataset_id!r} has an unreadable Pipeline 1 report: {exc}"
        ) from exc
    if report.get("status") != "success":
        raise Pipeline1PrivacyClearanceError(
            f"Dataset {dataset_id!r} Pipeline 1 status is not success."
        )
    if not report.get("validation", {}).get("passed"):
        raise Pipeline1PrivacyClearanceError(
            f"Dataset {dataset_id!r} failed Pipeline 1 post-clean validation."
        )

    source_name = report.get("original_dataset", {}).get("file_name")
    expected_source_name = Path(str(dataset["csv_file"])).name
    if source_name != expected_source_name:
        raise Pipeline1PrivacyClearanceError(
            f"Dataset {dataset_id!r} Pipeline 1 source mismatch: "
            f"expected {expected_source_name!r}, got {source_name!r}."
        )

    privacy = report.get("decision_summary", {}).get("privacy_actions")
    if not isinstance(privacy, dict):
        raise Pipeline1PrivacyClearanceError(
            f"Dataset {dataset_id!r} has no Pipeline 1 privacy audit."
        )
    if (
        privacy.get("reason") != "masked for privacy"
        or privacy.get("privacy_stage_completed") is not True
        or privacy.get("privacy_clearance_passed") is not True
    ):
        raise Pipeline1PrivacyClearanceError(
            f"Dataset {dataset_id!r} did not pass the Pipeline 1 privacy stage."
        )

    candidate_columns = set(privacy.get("pii_candidate_columns") or [])
    successful_mask_columns = {
        column
        for execution in report.get("executions", [])
        if execution.get("action_id") == "mask_column"
        and execution.get("status") == "success"
        and execution.get("audit_metadata", {}).get("reason")
        == "masked for privacy"
        for column in execution.get("target_columns", [])
    }
    if successful_mask_columns != candidate_columns:
        raise Pipeline1PrivacyClearanceError(
            f"Dataset {dataset_id!r} PII candidates do not match successful "
            "masking audit entries."
        )

    expected_digest = privacy.get("cleaned_dataset_sha256")
    actual_digest = hashlib.sha256(cleaned_path.read_bytes()).hexdigest()
    if not expected_digest or actual_digest != expected_digest:
        raise Pipeline1PrivacyClearanceError(
            f"Dataset {dataset_id!r} cleaned artifact failed privacy hash verification."
        )
    return pd.read_csv(cleaned_path, low_memory=False)


def _task_type(value: str) -> str:
    lowered = value.casefold()
    if "classification" in lowered:
        return "classification"
    if "regression" in lowered:
        return "regression"
    raise ValueError(f"Unsupported manifest task_type: {value!r}")


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _metadata_by_dataset(output_dir: Path, filename: str, value_key: str) -> dict[str, Any]:
    path = output_dir / filename
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(record["dataset_id"]): record.get(value_key)
        for record in payload.get("records", [])
    }


def _semantic_metadata(output_dir: Path) -> dict[str, Any]:
    result = {}
    profile_dir = output_dir / "dataset_semantic_profiles"
    if not profile_dir.is_dir():
        return result
    for path in profile_dir.glob("task_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            result[str(payload["dataset_id"])] = payload.get("semantic_profile")
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


def _save_run_artifacts(model_id, model, preprocessor, artifact_dir: Path) -> list[str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if model_id == "tabpfn":
        path = artifact_dir / "model.pkl"
        model.save_model(str(path))
        paths.append(str(path))
    elif model_id == "xtab":
        path = artifact_dir / "model.pt"
        model.predictor.save(path)
        paths.append(str(path))
    else:
        path = artifact_dir / "model.joblib"
        joblib.dump(model, path)
        paths.append(str(path))
        if preprocessor is not None:
            preprocessor_path = artifact_dir / "preprocessor.joblib"
            joblib.dump(preprocessor, preprocessor_path)
            paths.append(str(preprocessor_path))
    return paths


def _score(task_type, y_train, y_test, predictions):
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
    dataset, model_id, task_type, statistics, semantics, split_seed,
    run_id, base_model_id=None, execution_strategy="standard", split=None,
):
    regression = task_type == "regression"
    return {
        "dataset_id": str(dataset["task_id"]),
        "model_id": model_id,
        "base_model_id": base_model_id or model_id,
        "execution_strategy": execution_strategy,
        "run_id": run_id,
        "target_column": dataset["target_column"],
        "task_type": task_type,
        "feasible": True,
        "status": "failure",
        "outcome_reason": None,
        "infeasible_reason": None,
        "failure": None,
        "split": split or {
            "train_rows": None,
            "test_rows": None,
            "test_size": 0.2,
            "split_seed": split_seed,
            "stratified": task_type == "classification",
        },
        "metric_name": None,
        "metric_value": None,
        "baseline_relative_score": None,
        "train_runtime_seconds": None,
        "inference_runtime_seconds": None,
        "runtime_seconds": None,
        "oob_score": None,
        "oob_diagnostic_reason": None,
        "split_seed": split_seed,
        "hyperparams": {},
        "predictive_metrics": None,
        "calibration_metrics": {
            "brier_score": None,
            "ece": None,
            "ece_n_bins": None if regression else 15,
            "ece_method": None if regression else "top_label_equal_width",
            "status": "not_applicable" if regression else "unavailable",
            "reason": (
                "point-prediction regression protocol" if regression
                else "run did not produce validated probabilities"
            ),
        },
        "probability_metadata": None,
        "semantic_meta_features": semantics,
        "statistical_meta_features": statistics,
        "resource_usage": None,
        "source": "team-original-run",
    }


def _model_overrides(model_id: str, smoke: bool, random_forest_oob: bool):
    """Return only model-specific options supported by the selected wrapper."""
    if model_id == "random_forest":
        return {"oob_score": True} if random_forest_oob else {}
    if smoke and model_id == "xtab":
        return {"epochs": 1, "patience": 1}
    if smoke and model_id in {"realmlp", "tabm"}:
        overrides = {"n_epochs": 2}
        if model_id == "tabm":
            overrides["patience"] = 1
        return overrides
    if smoke and model_id == "tabiclv2":
        return {"n_estimators": 1}
    if smoke and model_id == "ft_transformer":
        return {
            "max_epochs": 2,
            "es_patience": 1,
            "use_checkpoints": False,
        }
    return {}


def _optional_oob_result(model_id: str, model, enabled: bool):
    """Read RF training-only OOB state without changing held-out scoring."""
    if model_id != "random_forest" or not enabled:
        return None, None
    estimator = getattr(model, "named_steps", {}).get("model")
    value = getattr(estimator, "oob_score_", None)
    if value is None:
        return None, "Random Forest OOB was enabled but oob_score_ is unavailable."
    return float(value), None


def _run_record_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record["dataset_id"]), str(record["model_id"]),
        str(record["execution_strategy"]), str(record["run_id"]),
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _write_log_records(output_dir: Path, records: list[dict[str, Any]]) -> None:
    """Atomically synchronize the human-readable CSV from JSON checkpoint rows."""
    for record in records:
        validate_log_record(record)
    path = output_dir / "model_run_log.csv"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({
                **record,
                "hyperparams": _json_cell(record["hyperparams"]),
                "predictive_metrics": _json_cell(record["predictive_metrics"]),
                "calibration_metrics": _json_cell(record["calibration_metrics"]),
                "probability_metadata": _json_cell(record["probability_metadata"]),
                "failure": _json_cell(record["failure"]),
                "split": _json_cell(record["split"]),
                "semantic_meta_features": _json_cell(record["semantic_meta_features"]),
                "statistical_meta_features": _json_cell(record["statistical_meta_features"]),
                "resource_usage": _json_cell(record["resource_usage"]),
            })
    temporary.replace(path)


def _release_model_resources() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prepare_dataset(
    dataset,
    split_seed,
    statistics_lookup,
    semantic_lookup,
    pipeline1_output_root,
):
    dataset_id = str(dataset["task_id"])
    task_type = _task_type(dataset["task_type"])
    frame = load_privacy_cleared_dataset(dataset, pipeline1_output_root)
    frame = frame.dropna(subset=[dataset["target_column"]]).reset_index(drop=True)
    X = frame.drop(columns=[dataset["target_column"]])
    y = frame[dataset["target_column"]]
    statistics = statistics_lookup.get(dataset_id) or statistical_meta_features(
        frame, dataset["target_column"]
    )
    semantics = semantic_lookup.get(dataset_id)
    class_count = int(y.nunique()) if task_type == "classification" else None
    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=split_seed, stratify=stratify
    )
    if task_type == "classification":
        target_encoder = LabelEncoder()
        y_train = target_encoder.fit_transform(y_train_raw.astype(str))
        y_test = target_encoder.transform(y_test_raw.astype(str))
    else:
        y_train = pd.to_numeric(y_train_raw, errors="raise").to_numpy()
        y_test = pd.to_numeric(y_test_raw, errors="raise").to_numpy()
    split = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": 0.2,
        "split_seed": split_seed,
        "stratified": task_type == "classification",
    }
    return (
        task_type, statistics, semantics, class_count,
        X_train, X_test, y_train, y_test, split,
    )


def run_evaluations(
    datasets: list[dict[str, Any]],
    output_dir: Path,
    pipeline1_output_root: Path,
    split_seed: int = 42,
    run_id: str | None = None,
    smoke: bool = False,
    random_forest_oob: bool = False,
    experimental_strategies: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Evaluate and immediately checkpoint every selected dataset/model pair."""
    output_dir = Path(output_dir)
    initialize_log(output_dir)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    statistics_lookup = _metadata_by_dataset(
        OUTPUT_DIR, "statistical_meta_features.json", "statistical_meta_features"
    )
    semantic_lookup = _semantic_metadata(OUTPUT_DIR)
    unknown_strategies = set(experimental_strategies) - set(EXPERIMENTAL_EXECUTION_STRATEGIES)
    if unknown_strategies:
        raise ValueError(f"Unknown experimental strategies: {sorted(unknown_strategies)}")
    run_specs = [
        (model_id, model_id, "standard") for model_id in EVALUATION_MODEL_IDS
    ] + [
        (
            variant_id,
            EXPERIMENTAL_EXECUTION_STRATEGIES[variant_id]["base_model_id"],
            "crumb",
        )
        for variant_id in experimental_strategies
    ]

    checkpoint_path = output_dir / "runs" / run_id / "model_run_rows.json"
    if checkpoint_path.is_file():
        rows = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Invalid benchmark checkpoint: {checkpoint_path}")
        for row in rows:
            validate_log_record(row)
    else:
        rows = []
    _write_log_records(output_dir, rows)
    completed_keys = {_run_record_key(row) for row in rows}
    selected_keys = {
        (str(dataset["task_id"]), model_id, strategy, run_id)
        for dataset in datasets for model_id, _base_model_id, strategy in run_specs
    }
    total = len(selected_keys)

    def persist(record: dict[str, Any], dataset_index: int, model_index: int) -> None:
        key = _run_record_key(record)
        if key in completed_keys:
            return
        rows.append(record)
        completed_keys.add(key)
        _atomic_json(checkpoint_path, rows)
        _atomic_json(output_dir / "model_run_rows.json", rows)
        _write_log_records(output_dir, rows)
        completed = len(completed_keys & selected_keys)
        metric = (
            f" | {record['metric_name']}={record['metric_value']:.6g}"
            if record["metric_value"] is not None else ""
        )
        print(
            f"Dataset {dataset_index}/{len(datasets)} | Model {model_index}/{len(run_specs)} | "
            f"{record['model_id']} | {record['status']}{metric}", flush=True,
        )
        statuses = [row["status"] for row in rows if _run_record_key(row) in selected_keys]
        print(
            f"Completed {completed}/{total} | success={statuses.count('success')} | "
            f"infeasible={statuses.count('infeasible_skip')} | failure={statuses.count('failure')}",
            flush=True,
        )

    for dataset_index, dataset in enumerate(datasets, 1):
        dataset_id = str(dataset["task_id"])
        pending_specs = [
            spec for spec in run_specs
            if (dataset_id, spec[0], spec[2], run_id) not in completed_keys
        ]
        if not pending_specs:
            print(f"[SKIP] Dataset {dataset_index}/{len(datasets)} task_{dataset_id} complete", flush=True)
            continue
        try:
            (
                task_type, statistics, semantics, class_count,
                X_train, X_test, y_train, y_test, split,
            ) = _prepare_dataset(
                dataset,
                split_seed,
                statistics_lookup,
                semantic_lookup,
                pipeline1_output_root,
            )
        except Exception as exc:
            task_type = _task_type(dataset["task_type"])
            for model_id, base_model_id, execution_strategy in pending_specs:
                record = _base_record(
                    dataset, model_id, task_type,
                    statistics_lookup.get(dataset_id), semantic_lookup.get(dataset_id),
                    split_seed, run_id, base_model_id, execution_strategy,
                )
                record["outcome_reason"] = f"dataset_setup: {type(exc).__name__}: {exc}"
                record["failure"] = {
                    "stage": "dataset_setup",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
                model_index = next(i for i, spec in enumerate(run_specs, 1) if spec == (model_id, base_model_id, execution_strategy))
                persist(record, dataset_index, model_index)
            continue

        for model_id, base_model_id, execution_strategy in pending_specs:
            model_index = next(i for i, spec in enumerate(run_specs, 1) if spec == (model_id, base_model_id, execution_strategy))
            record = _base_record(
                dataset, model_id, task_type, statistics, semantics, split_seed,
                run_id, base_model_id, execution_strategy, split,
            )
            if execution_strategy == "crumb":
                result = crumb_feasibility(
                    base_model_id, len(X_train), X_train.shape[1], task_type, class_count
                )
                allowed, reason = result["crumb_variant_feasible"], result["reason"]
            else:
                allowed, reason = feasibility(
                    model_id, len(X_train), X_train.shape[1], task_type, class_count
                )
            if allowed is False:
                record.update({
                    "feasible": False,
                    "status": "infeasible_skip",
                    "outcome_reason": reason,
                    "infeasible_reason": reason,
                })
                persist(record, dataset_index, model_index)
                continue
            if allowed is None:
                record["outcome_reason"] = f"unresolved_feasibility: {reason}"
                record["failure"] = {
                    "stage": "feasibility",
                    "exception_type": "UnresolvedFeasibilityError",
                    "message": str(reason),
                }
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
                overrides = _model_overrides(
                    model_id, smoke, random_forest_oob
                )
                model = package.build_model(task_type, split_seed, **overrides)
                package.train_model(model, X_train_model, y_train)
                oob_score, oob_reason = _optional_oob_result(
                    model_id, model, random_forest_oob
                )
                train_runtime = time.perf_counter() - train_started
                inference_started = time.perf_counter()
                X_test_model = (
                    preprocessor.transform(X_test) if preprocessor is not None else X_test.copy()
                )
                predictions = np.asarray(package.predict_model(model, X_test_model)).reshape(-1)
                if task_type == "classification":
                    predictions = pd.to_numeric(
                        pd.Series(predictions), errors="raise"
                    ).to_numpy(dtype=np.asarray(y_test).dtype)
                    predictive_metrics, calibration_metrics, probability_metadata = (
                        classification_results(
                            y_train, y_test, predictions, model_id, model, X_test_model
                        )
                    )
                else:
                    predictive_metrics, calibration_metrics, probability_metadata = (
                        regression_results(y_test, predictions)
                    )
                inference_runtime = time.perf_counter() - inference_started
                metric_name, metric_value, relative = _score(
                    task_type, y_train, y_test, predictions
                )
                artifact_dir = training_artifact_dir(
                    model_id, dataset_id, run_id, output_dir / "run_artifacts"
                )
                artifact_paths = _save_run_artifacts(
                    model_id, model, preprocessor, artifact_dir
                )
                record.update({
                    "status": "success",
                    "outcome_reason": f"artifacts={artifact_paths}",
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "baseline_relative_score": relative,
                    "train_runtime_seconds": train_runtime,
                    "inference_runtime_seconds": inference_runtime,
                    "runtime_seconds": train_runtime + inference_runtime,
                    "oob_score": oob_score,
                    "oob_diagnostic_reason": oob_reason,
                    "hyperparams": {"model_overrides": overrides},
                    "predictive_metrics": predictive_metrics,
                    "calibration_metrics": calibration_metrics,
                    "probability_metadata": probability_metadata,
                })
            except Exception as exc:
                record["outcome_reason"] = f"{type(exc).__name__}: {exc}"
                record["failure"] = {
                    "stage": "train_or_evaluate",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            persist(record, dataset_index, model_index)
            del model, preprocessor
            _release_model_resources()
    return rows


from benchmarks.tabarena.tabarena_benchmark import (  # noqa: E402
    CANONICAL_MODEL_IDS,
    PERFORMANCE_PATH,
    load_benchmark_datasets,
    run_evaluations as run_tabarena_evaluations,
)

# Public execution entry point: direct local TabArena CSVs, no Pipeline 1 gate.
run_evaluations = run_tabarena_evaluations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", help="Task id or exact dataset name")
    parser.add_argument("--all", action="store_true", help="Explicitly authorize the full manifest")
    parser.add_argument("--output", type=Path, default=PERFORMANCE_PATH)
    parser.add_argument("--model", action="append", choices=CANONICAL_MODEL_IDS)
    parser.add_argument("--retry-runtime-failures", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    if not args.all and not args.dataset:
        parser.error("Supply at least one --dataset, or explicitly use --all")
    if args.all and args.dataset:
        parser.error("Use either --all or --dataset, not both")
    manifest = load_benchmark_datasets()
    if args.all:
        selected = manifest
    else:
        selectors = set(args.dataset)
        selected = [
            item for item in manifest
            if str(item["task_id"]) in selectors or item["dataset_name"] in selectors
        ]
        found = {str(item["task_id"]) for item in selected} | {
            item["dataset_name"] for item in selected
        }
        missing = selectors - found
        if missing:
            parser.error(f"Unknown dataset selectors: {sorted(missing)}")
    model_ids = tuple(args.model) if args.model else CANONICAL_MODEL_IDS
    rows = run_tabarena_evaluations(
        selected, args.output, args.split_seed, model_ids,
        args.retry_runtime_failures,
    )
    selected_keys = {
        (str(item["task_id"]), model_id)
        for item in selected
        for model_id in model_ids
    }
    persisted = sum(
        (row["dataset_id"], row["model_id"]) in selected_keys for row in rows
    )
    print(json.dumps({
        "selected_combinations": len(selected_keys),
        "persisted_selected_records": persisted,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
