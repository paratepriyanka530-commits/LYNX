"""Executable, auditable Pipeline 2 dataset-by-model evaluation runner.

The CLI requires explicit dataset selectors unless ``--all`` is supplied.  This
prevents an accidental full TabArena sweep while still providing the full loop.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from benchmarks.tabarena.pipeline2_groundwork import (
    EVALUATION_MODEL_IDS,
    LOG_COLUMNS,
    OUTPUT_DIR,
    feasibility,
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
    profile_dir = output_dir / "semantic_profiles"
    if not profile_dir.is_dir():
        return result
    for path in profile_dir.glob("task_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[str(payload["dataset_id"])] = payload.get("semantic_meta_features")
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


def _base_record(dataset, model_id, task_type, statistics, semantics, split_seed):
    return {
        "dataset_id": str(dataset["task_id"]),
        "model_id": model_id,
        "target_column": dataset["target_column"],
        "task_type": task_type,
        "feasible": True,
        "status": "failure",
        "outcome_reason": None,
        "metric_name": None,
        "metric_value": None,
        "baseline_relative_score": None,
        "train_runtime_seconds": None,
        "inference_runtime_seconds": None,
        "runtime_seconds": None,
        "split_seed": split_seed,
        "hyperparams": {},
        "semantic_meta_features": semantics,
        "statistical_meta_features": statistics,
        "source": "team-original-run",
    }


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
    return task_type, statistics, semantics, class_count, X_train, X_test, y_train, y_test


def run_evaluations(
    datasets: list[dict[str, Any]],
    output_dir: Path,
    pipeline1_output_root: Path,
    split_seed: int = 42,
    run_id: str | None = None,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate every selected dataset/model pair and log every outcome."""
    output_dir = Path(output_dir)
    initialize_log(output_dir)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    statistics_lookup = _metadata_by_dataset(
        OUTPUT_DIR, "statistical_meta_features.json", "statistical_meta_features"
    )
    semantic_lookup = _semantic_metadata(OUTPUT_DIR)
    rows = []

    for dataset in datasets:
        dataset_id = str(dataset["task_id"])
        try:
            (
                task_type, statistics, semantics, class_count,
                X_train, X_test, y_train, y_test,
            ) = _prepare_dataset(
                dataset,
                split_seed,
                statistics_lookup,
                semantic_lookup,
                pipeline1_output_root,
            )
        except Exception as exc:
            task_type = _task_type(dataset["task_type"])
            for model_id in EVALUATION_MODEL_IDS:
                record = _base_record(
                    dataset, model_id, task_type,
                    statistics_lookup.get(dataset_id), semantic_lookup.get(dataset_id),
                    split_seed,
                )
                record["outcome_reason"] = f"dataset_setup: {type(exc).__name__}: {exc}"
                rows.append(record)
            continue

        for model_id in EVALUATION_MODEL_IDS:
            record = _base_record(
                dataset, model_id, task_type, statistics, semantics, split_seed
            )
            allowed, reason = feasibility(
                model_id, len(X_train), X_train.shape[1], task_type, class_count
            )
            if allowed is False:
                record.update({
                    "feasible": False,
                    "status": "infeasible_skip",
                    "outcome_reason": reason,
                })
                rows.append(record)
                continue
            if allowed is None:
                record["outcome_reason"] = f"unresolved_feasibility: {reason}"
                rows.append(record)
                continue

            try:
                package = importlib.import_module(
                    f"pipelines.pipeline2_prediction.models.{model_id}"
                )
                train_started = time.perf_counter()
                X_train_model, preprocessor = _fit_feature_preprocessor(model_id, X_train)
                if smoke and model_id == "xtab":
                    overrides = {"epochs": 1, "patience": 1}
                elif smoke and model_id in {"realmlp", "tabm"}:
                    overrides = {"n_epochs": 2}
                    if model_id == "tabm":
                        overrides["patience"] = 1
                elif smoke and model_id == "tabiclv2":
                    overrides = {"n_estimators": 1}
                elif smoke and model_id == "ft_transformer":
                    overrides = {
                        "max_epochs": 2,
                        "es_patience": 1,
                        "use_checkpoints": False,
                    }
                else:
                    overrides = {}
                model = package.build_model(task_type, split_seed, **overrides)
                package.train_model(model, X_train_model, y_train)
                train_runtime = time.perf_counter() - train_started
                inference_started = time.perf_counter()
                X_test_model = (
                    preprocessor.transform(X_test) if preprocessor is not None else X_test.copy()
                )
                predictions = np.asarray(package.predict_model(model, X_test_model)).reshape(-1)
                inference_runtime = time.perf_counter() - inference_started
                if task_type == "classification":
                    predictions = pd.to_numeric(
                        pd.Series(predictions), errors="raise"
                    ).to_numpy(dtype=np.asarray(y_test).dtype)
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
                    "hyperparams": {"smoke_overrides": overrides},
                })
            except Exception as exc:
                record["outcome_reason"] = f"{type(exc).__name__}: {exc}"
            rows.append(record)

    log_path = output_dir / "model_run_log.csv"
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        for record in rows:
            validate_log_record(record)
            writer.writerow({
                **record,
                "hyperparams": _json_cell(record["hyperparams"]),
                "semantic_meta_features": _json_cell(record["semantic_meta_features"]),
                "statistical_meta_features": _json_cell(record["statistical_meta_features"]),
            })
    (output_dir / "model_run_rows.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", help="Task id or exact dataset name")
    parser.add_argument("--all", action="store_true", help="Explicitly authorize the full manifest")
    parser.add_argument("--manifest-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pipeline1-output-root",
        type=Path,
        required=True,
        help=(
            "Root containing one privacy-cleared Pipeline 1 artifact directory "
            "per task_id"
        ),
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true", help="Use one XTab epoch for a lifecycle dry run")
    args = parser.parse_args()
    if not args.all and not args.dataset:
        parser.error("Supply at least one --dataset, or explicitly use --all")
    if args.all and args.dataset:
        parser.error("Use either --all or --dataset, not both")
    manifest = load_manifest(args.manifest_dir)
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
    rows = run_evaluations(
        selected,
        args.output_dir,
        args.pipeline1_output_root,
        args.split_seed,
        args.run_id,
        args.smoke,
    )
    print(json.dumps(rows, indent=2, default=str))
    return 0 if all(row["status"] != "failure" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
