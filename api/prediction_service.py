"""Durable asynchronous prediction jobs for the application API.

The worker is deliberately an adapter: task resolution and prediction are
delegated to the existing resolver and production orchestrator. Their heavy
imports occur only inside a worker thread.

Startup reconciliation conservatively fails abandoned queued and running jobs.
Neither state is safe to replay without a cross-process claim protocol or an
ML checkpoint, so users can explicitly request a new prediction instead.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from io import StringIO
from datetime import datetime, timezone
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
from threading import RLock
import time
from typing import Any, Callable, Iterator, Mapping
from uuid import UUID, uuid4

from api.run_service import ApplicationRunService, RunStore, ServiceError


PREDICTION_STATES = {"queued", "running", "complete", "failed"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = PROJECT_ROOT / "pipelines/pipeline2_prediction/model_selector/lynx_model_selector.joblib"
SELECTOR_METADATA_PATH = SELECTOR_PATH.with_name("final_selector_metadata.json")
EXPECTED_SELECTOR_SHA256 = "46ff89a509316389aa23b0f4b8083f2f83e8804759a4966e83e4616d59e06550"
EXPECTED_SELECTOR_METADATA_SHA256 = "c096a62acefc7189b7d0e6ea2d443cf7b5b56d83e8a56cb873af07dece1abe03"
logger = logging.getLogger(__name__)

Resolver = Callable[[Any, str, Mapping[str, Any]], Any]
Orchestrator = Callable[..., Any]
DEFAULT_PREDICTION_ROW_LIMIT = 50
MAX_PREDICTION_ROW_LIMIT = 200
INTERRUPTED_MESSAGE = "The prediction was interrupted before it completed. Please run the prediction again."


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_resolver(dataframe: Any, target_column: str, report: Mapping[str, Any]) -> Any:
    resolver = importlib.import_module("api.task_type_resolver").resolve_task_type
    return resolver(dataframe, target_column, report)


def _default_orchestrator(dataframe: Any, **kwargs: Any) -> Any:
    module = importlib.import_module(
        "pipelines.pipeline2_prediction.orchestration.prediction_orchestrator"
    )
    return module.orchestrate_prediction(dataframe, **kwargs)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _public_prediction_row(row: list[str], task_type: str) -> dict[str, Any]:
    """The single Stage 4D/5B definition of a public prediction row."""
    if task_type == "classification":
        probabilities = [_finite_number(value) for value in row[2:]]
        usable = bool(probabilities) and all(
            value is not None and 0.0 <= value <= 1.0 for value in probabilities
        )
        return {
            "actual": row[0] or None,
            "prediction": row[1] or None,
            "confidence": max(probabilities) if usable else None,
        }
    actual = _finite_number(row[0])
    prediction = _finite_number(row[1])
    return {
        "actual": actual,
        "prediction": prediction,
        "absolute_error": (
            _finite_number(abs(actual - prediction))
            if actual is not None and prediction is not None else None
        ),
    }


def _public_csv_cell(value: Any) -> str:
    """Neutralize spreadsheet formulas in text, never in numeric cells."""
    if value is None:
        return ""
    if isinstance(value, str):
        return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value
    return str(value)


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _metric(mapping: Mapping[str, Any], key: str) -> int | float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServiceError(
            500, "prediction_result_invalid", "Prediction result is invalid."
        )
    if not math.isfinite(float(value)):
        raise ServiceError(
            500, "prediction_result_invalid", "Prediction result is invalid."
        )
    return value


def project_frontend_result(internal: Mapping[str, Any]) -> dict[str, Any]:
    """Return the strict frontend whitelist for one successful internal result."""
    task_type = internal.get("task_type")
    if task_type not in {"classification", "regression"}:
        raise ServiceError(
            500, "prediction_result_invalid", "Prediction result is invalid."
        )
    target = internal.get("target_column")
    prediction_id = internal.get("prediction_id")
    if not isinstance(target, str) or not isinstance(prediction_id, str):
        raise ServiceError(
            500, "prediction_result_invalid", "Prediction result is invalid."
        )

    resolution = _mapping(internal.get("task_resolution"))
    evidence = _mapping(resolution.get("evidence"))
    audit = _mapping(internal.get("orchestrator_audit"))
    dataset_audit = _mapping(audit.get("dataset"))
    split_audit = _mapping(audit.get("split"))
    feature_names = dataset_audit.get("feature_names")
    feature_count = len(feature_names) if isinstance(feature_names, list) else None

    evaluation = _mapping(internal.get("evaluation"))
    predictive = _mapping(evaluation.get("predictive_metrics"))
    if task_type == "classification":
        primary_keys = ("balanced_accuracy", "f1_macro", "roc_auc", "accuracy")
        secondary_keys = ("precision_macro", "recall_macro", "mcc", "log_loss")
        if any(key not in predictive for key in primary_keys):
            raise ServiceError(
                500, "prediction_result_invalid", "Prediction result is invalid."
            )
        metrics: dict[str, Any] = {
            "primary": {key: _metric(predictive, key) for key in primary_keys},
            "secondary": {
                key: _metric(predictive, key) for key in secondary_keys
                if key in predictive
            },
            "confusion_matrix": predictive.get("confusion_matrix"),
        }
        confusion = metrics["confusion_matrix"]
        if confusion is not None and not (
            isinstance(confusion, list)
            and all(
                isinstance(row, list)
                and all(isinstance(value, int) and not isinstance(value, bool) for value in row)
                for row in confusion
            )
        ):
            raise ServiceError(
                500, "prediction_result_invalid", "Prediction result is invalid."
            )
        calibration = _mapping(evaluation.get("calibration_metrics"))
        if calibration:
            safe_calibration_keys = (
                "brier_score", "ece", "ece_n_bins", "ece_method", "status", "reason",
            )
            metrics["calibration"] = {}
            for key in safe_calibration_keys:
                if key not in calibration:
                    continue
                value = calibration[key]
                if key in {"brier_score", "ece"}:
                    value = _metric(calibration, key)
                elif key == "ece_n_bins" and value is not None:
                    value = _integer(value)
                    if value is None:
                        raise ServiceError(
                            500, "prediction_result_invalid",
                            "Prediction result is invalid.",
                        )
                elif key in {"ece_method", "status", "reason"} and value is not None:
                    if not isinstance(value, str):
                        raise ServiceError(
                            500, "prediction_result_invalid",
                            "Prediction result is invalid.",
                        )
                metrics["calibration"][key] = value
    else:
        regression_keys = ("rmse", "mae", "r2", "nrmse")
        if any(key not in predictive for key in regression_keys):
            raise ServiceError(
                500, "prediction_result_invalid", "Prediction result is invalid."
            )
        metrics = {key: _metric(predictive, key) for key in regression_keys}
        if "nrmse_normalizer" in predictive:
            normalizer = predictive["nrmse_normalizer"]
            if normalizer is not None and not isinstance(normalizer, str):
                raise ServiceError(
                    500, "prediction_result_invalid", "Prediction result is invalid."
                )
            metrics["nrmse_normalizer"] = normalizer

    summary = _mapping(internal.get("prediction_summary"))
    row_count = _integer(summary.get("row_count"))
    probabilities_available = bool(summary.get("probabilities_available"))
    runtime = _mapping(internal.get("runtime"))
    return {
        "status": "complete",
        "prediction_id": prediction_id,
        "target": target,
        "task_type": task_type,
        "task_resolution": {
            "status": "resolved",
            "task_type": task_type,
            "explanation": f"Resolved as a {task_type} target.",
        },
        "dataset": {
            "name": dataset_audit.get("dataset_name"),
            "total_rows": _integer(dataset_audit.get("row_count")),
            "training_rows": _integer(split_audit.get("train_row_count")),
            "test_rows": _integer(split_audit.get("test_row_count")),
            "feature_count": feature_count,
            "target_missing_count": _integer(evidence.get("missing_count")),
        },
        "metrics": metrics,
        "predictions": {
            "available": row_count is not None,
            "row_count": row_count,
            "has_probabilities": probabilities_available,
            "confidence_available": bool(
                task_type == "classification" and probabilities_available
            ),
        },
        "runtime": {
            "train_runtime_seconds": _metric(runtime, "train_runtime_seconds"),
            "inference_runtime_seconds": _metric(
                runtime, "inference_runtime_seconds"
            ),
        },
    }


class PredictionExecutionFailure(RuntimeError):
    """Controlled worker failure persisted without the underlying exception."""

    def __init__(self, code: str, message: str, stage: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage


class PredictionStore:
    """Atomic prediction state/result storage beneath an application run."""

    def __init__(self, run_store: RunStore):
        self.run_store = run_store
        self._locks_guard = RLock()
        self._prediction_locks: dict[tuple[str, str], RLock] = {}

    @staticmethod
    def validate_prediction_id(prediction_id: str) -> str:
        try:
            parsed = UUID(str(prediction_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ServiceError(400, "malformed_prediction_id", "Prediction ID must be a valid UUID.") from exc
        return str(parsed)

    def _prediction_lock(self, run_id: str, prediction_id: str) -> RLock:
        run_id = self.run_store.validate_run_id(run_id)
        prediction_id = self.validate_prediction_id(prediction_id)
        key = (run_id, prediction_id)
        with self._locks_guard:
            return self._prediction_locks.setdefault(key, RLock())

    def _paths(self, run_id: str, prediction_id: str, *, require_exists: bool = True) -> dict[str, Path]:
        run_id = self.run_store.validate_run_id(run_id)
        prediction_id = self.validate_prediction_id(prediction_id)
        run_directory = self.run_store.run_directory(run_id)
        root = (run_directory / "predictions").resolve()
        directory = (root / prediction_id).resolve()
        try:
            root.relative_to(run_directory)
            directory.relative_to(root)
        except ValueError as exc:
            raise ServiceError(400, "invalid_prediction_path", "Resolved prediction path is outside its application run.") from exc
        if require_exists and not directory.is_dir():
            raise ServiceError(404, "prediction_not_found", "The requested prediction does not exist for this run.")
        return {
            "root": root, "prediction": directory,
            "state": directory / "state.json", "result": directory / "result.json",
            "predictions": directory / "predictions.csv",
        }

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def create(self, run_id: str, target_column: str) -> dict[str, Any]:
        run_id = self.run_store.validate_run_id(run_id)
        with self.run_store.locked_run(run_id):
            self.run_store.run_directory(run_id)
            for _attempt in range(5):
                prediction_id = str(uuid4())
                paths = self._paths(run_id, prediction_id, require_exists=False)
                try:
                    paths["root"].mkdir(exist_ok=True)
                    paths["prediction"].mkdir(exist_ok=False)
                except FileExistsError:
                    continue
                now = _utc_now()
                state = {
                    "prediction_id": prediction_id, "run_id": run_id,
                    "state": "queued", "stage": "queued", "target_column": target_column,
                    "created_at": now, "updated_at": now, "error": None,
                }
                with self._prediction_lock(run_id, prediction_id):
                    self._atomic_json(paths["state"], state)
                return state
        raise ServiceError(500, "prediction_id_unavailable", "A prediction ID could not be allocated.")

    def read(self, run_id: str, prediction_id: str) -> dict[str, Any]:
        with self._prediction_lock(run_id, prediction_id):
            path = self._paths(run_id, prediction_id)["state"]
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise ServiceError(404, "prediction_not_found", "The requested prediction does not exist for this run.") from exc
            except (OSError, json.JSONDecodeError) as exc:
                raise ServiceError(500, "prediction_state_unavailable", "Prediction state is unavailable.") from exc

    def update(self, run_id: str, prediction_id: str, **changes: Any) -> dict[str, Any]:
        immutable = {"prediction_id", "run_id", "target_column", "created_at"}
        if immutable.intersection(changes):
            raise ValueError("Prediction identity fields cannot be changed")
        if "state" in changes and changes["state"] not in PREDICTION_STATES:
            raise ValueError("Unsupported prediction state")
        with self._prediction_lock(run_id, prediction_id):
            paths = self._paths(run_id, prediction_id)
            try:
                state = json.loads(paths["state"].read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise ServiceError(404, "prediction_not_found", "The requested prediction does not exist for this run.") from exc
            except (OSError, json.JSONDecodeError) as exc:
                raise ServiceError(500, "prediction_state_unavailable", "Prediction state is unavailable.") from exc
            state.update(changes)
            state["updated_at"] = _utc_now()
            self._atomic_json(paths["state"], state)
            return state

    def transition_if(
        self, run_id: str, prediction_id: str, *, expected: set[str],
        owner: str | None = None, **changes: Any,
    ) -> dict[str, Any] | None:
        """Atomically change non-final state only while the expected owner holds it."""
        immutable = {"prediction_id", "run_id", "target_column", "created_at"}
        if immutable.intersection(changes):
            raise ValueError("Prediction identity fields cannot be changed")
        if "state" in changes and changes["state"] not in PREDICTION_STATES:
            raise ValueError("Unsupported prediction state")
        with self._prediction_lock(run_id, prediction_id):
            current = self.read(run_id, prediction_id)
            if current.get("state") not in expected:
                return None
            if owner is not None and current.get("worker_instance_id") != owner:
                return None
            current.update(changes)
            current["updated_at"] = _utc_now()
            self._atomic_json(self._paths(run_id, prediction_id)["state"], current)
            return current

    def write_result(self, run_id: str, prediction_id: str, result: Mapping[str, Any]) -> None:
        with self._prediction_lock(run_id, prediction_id):
            self._atomic_json(self._paths(run_id, prediction_id)["result"], result)

    def write_prediction_rows(self, run_id: str, prediction_id: str, fieldnames: list[str], rows: list[list[Any]]) -> None:
        """Atomically persist potentially large row-level prediction output."""
        with self._prediction_lock(run_id, prediction_id):
            path = self._paths(run_id, prediction_id)["predictions"]
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                with temporary.open("x", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(fieldnames)
                    writer.writerows(rows)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def read_result(self, run_id: str, prediction_id: str) -> dict[str, Any]:
        with self._prediction_lock(run_id, prediction_id):
            try:
                payload = json.loads(self._paths(run_id, prediction_id)["result"].read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
                raise ServiceError(500, "prediction_result_unavailable", "Prediction result is unavailable.") from exc
            if not isinstance(payload, dict):
                raise ServiceError(500, "prediction_result_unavailable", "Prediction result is unavailable.")
            return payload


class PredictionJobService:
    """Create and asynchronously execute run-scoped prediction jobs."""

    def __init__(
        self, run_service: ApplicationRunService, store: PredictionStore | None = None, *,
        resolver: Resolver = _default_resolver, orchestrator: Orchestrator = _default_orchestrator,
        auto_execute: bool = True, worker_count: int | None = None,
        selector_path: Path = SELECTOR_PATH, selector_metadata_path: Path = SELECTOR_METADATA_PATH,
    ):
        self.run_service = run_service
        self.store = store or PredictionStore(run_service.store)
        self.resolver = resolver
        self.orchestrator = orchestrator
        self.selector_path = Path(selector_path)
        self.selector_metadata_path = Path(selector_metadata_path)
        self.instance_id = str(uuid4())
        self._schedule_lock = RLock()
        self._active_predictions: set[tuple[str, str]] = set()
        self._created_predictions: set[tuple[str, str]] = set()
        self._shutting_down = False
        configured = worker_count or run_service.settings.prediction_worker_count
        self.executor = ThreadPoolExecutor(max_workers=configured, thread_name_prefix="lynx-prediction") if auto_execute else None

    def _schedule(self, run_id: str, prediction_id: str) -> bool:
        """Submit one prediction ID at most once in this service instance."""
        if self.executor is None:
            return False
        key = (run_id, prediction_id)
        with self._schedule_lock:
            if self._shutting_down:
                self.store.transition_if(
                    run_id, prediction_id, expected={"queued"}, state="failed",
                    stage="interrupted", error={"code": "prediction_interrupted", "message": INTERRUPTED_MESSAGE},
                    interrupted_at=_utc_now(),
                )
                return False
            if key in self._active_predictions:
                return False
            self._active_predictions.add(key)
            try:
                future = self.executor.submit(self._execute, run_id, prediction_id)
            except RuntimeError:
                self._active_predictions.remove(key)
                self.store.transition_if(
                    run_id, prediction_id, expected={"queued"}, state="failed",
                    stage="interrupted", error={"code": "prediction_interrupted", "message": INTERRUPTED_MESSAGE},
                )
                return False
            future.add_done_callback(lambda _future: self._discard_active(key))
            return True

    def _discard_active(self, key: tuple[str, str]) -> None:
        with self._schedule_lock:
            self._active_predictions.discard(key)

    def reconcile_startup(self) -> dict[str, int]:
        """Fail orphaned non-final jobs without importing or replaying ML code."""
        summary = {"queued_interrupted": 0, "running_interrupted": 0, "invalid_entries": 0}
        root = self.run_service.store.root
        with self._schedule_lock:
            try:
                runs = tuple(root.iterdir())
            except OSError:
                logger.warning("Prediction recovery could not scan the runtime root")
                return summary
            for run_dir in runs:
                if not run_dir.is_dir() or run_dir.is_symlink():
                    continue
                try:
                    run_id = self.run_service.store.validate_run_id(run_dir.name)
                    if run_dir.resolve() != self.run_service.store.run_directory(run_id):
                        continue
                    prediction_root = run_dir / "predictions"
                    if not prediction_root.is_dir() or prediction_root.is_symlink():
                        continue
                    predictions = tuple(prediction_root.iterdir())
                except (ServiceError, OSError):
                    summary["invalid_entries"] += 1
                    continue
                for prediction_dir in predictions:
                    if not prediction_dir.is_dir() or prediction_dir.is_symlink():
                        continue
                    try:
                        prediction_id = self.store.validate_prediction_id(prediction_dir.name)
                        if prediction_dir.resolve() != self.store._paths(run_id, prediction_id)["prediction"]:
                            continue
                        state = self.store.read(run_id, prediction_id)
                        if not isinstance(state, dict) or state.get("run_id") != run_id or state.get("prediction_id") != prediction_id:
                            summary["invalid_entries"] += 1
                            continue
                        status = state.get("state")
                        if status == "complete":
                            paths = self.store._paths(run_id, prediction_id)
                            if not all(paths[name].is_file() and not paths[name].is_symlink() for name in ("result", "predictions")):
                                summary["invalid_entries"] += 1
                            else:
                                # Reuse the existing frontend-safe result validator;
                                # never mutate or replay a complete job.
                                project_frontend_result(self.store.read_result(run_id, prediction_id))
                            continue
                        if status == "failed":
                            continue
                        if status not in {"queued", "running"}:
                            summary["invalid_entries"] += 1
                            continue
                        if status == "queued" and (run_id, prediction_id) in self._created_predictions:
                            continue
                        if status == "running" and state.get("worker_instance_id") == self.instance_id:
                            continue
                        if (run_id, prediction_id) in self._active_predictions:
                            continue
                        changed = self.store.transition_if(
                            run_id, prediction_id, expected={status}, state="failed",
                            stage="interrupted", error={"code": "prediction_interrupted", "message": INTERRUPTED_MESSAGE},
                            interrupted_at=_utc_now(),
                        )
                        if changed is not None:
                            summary[f"{status}_interrupted"] += 1
                    except Exception:
                        # A single malformed/incomplete entry cannot block API startup.
                        # Do not log the entry or exception: both may contain local paths.
                        summary["invalid_entries"] += 1
                        continue
        logger.info(
            "Prediction recovery: %d queued, %d running interrupted; %d invalid entries skipped",
            summary["queued_interrupted"], summary["running_interrupted"], summary["invalid_entries"],
        )
        return summary

    def create(self, run_id: str, target_column: str) -> dict[str, Any]:
        with self._schedule_lock:
            if self._shutting_down:
                raise ServiceError(503, "prediction_service_stopping", "Prediction service is stopping. Please try again later.")
        run_id = self.run_service.store.validate_run_id(run_id)
        if not isinstance(target_column, str) or not target_column.strip():
            raise ServiceError(422, "invalid_target", "Target column must be a non-empty string.")
        with self.run_service.store.locked_run(run_id):
            context = self.run_service.prediction_context(run_id)
            if not context["eligibility"]["eligible"]:
                raise ServiceError(409, "prediction_not_eligible", "The prepared dataset is not eligible for prediction.")
            if context["target_columns"].count(target_column) != 1:
                raise ServiceError(422, "target_not_found", "Target column was not found exactly once in the prepared dataset.")
            state = self.store.create(run_id, target_column)
            with self._schedule_lock:
                self._created_predictions.add((run_id, state["prediction_id"]))
        self._schedule(run_id, state["prediction_id"])
        return {"run_id": run_id, "prediction_id": state["prediction_id"], "state": "queued", "status_url": f"/api/runs/{run_id}/predictions/{state['prediction_id']}"}

    @staticmethod
    def _resolution_dict(resolution: Any) -> dict[str, Any]:
        converter = getattr(resolution, "to_dict", None)
        payload = converter() if callable(converter) else resolution
        if not isinstance(payload, Mapping):
            raise PredictionExecutionFailure("task_resolution_invalid", "Task resolution returned an invalid result.", "task_resolution")
        return dict(payload)

    @staticmethod
    def _semantic_profile(report: Mapping[str, Any], target: str) -> dict[str, Any]:
        """Conservatively adapt existing Pipeline 1 semantics to the selector schema."""
        profile = _mapping(report.get("semantic_profile"))
        context = _mapping(profile.get("dataset_context"))
        columns = [item for item in (profile.get("columns") or []) if isinstance(item, Mapping)]
        target_record = next((dict(item) for item in columns if item.get("column_name") == target), {})

        def text_value(value: Any, fallback: str = "unknown") -> str:
            return value.strip() if isinstance(value, str) and value.strip() else fallback

        feature_groups: list[str] = []
        for item in columns:
            if item.get("column_name") == target:
                continue
            group = item.get("semantic_type") or item.get("semantic_role")
            if isinstance(group, str) and group.strip() and group.strip() not in feature_groups:
                feature_groups.append(group.strip())
            if len(feature_groups) == 12:
                break
        description = text_value(context.get("dataset_description"))
        confidence_value = context.get("domain_confidence", context.get("confidence"))
        try:
            confidence = min(1.0, max(0.0, float(confidence_value)))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "probable_domain": text_value(context.get("probable_domain")),
            "subdomain": text_value(context.get("subdomain")),
            "dataset_purpose": text_value(context.get("dataset_purpose"), description),
            "row_entity": text_value(context.get("row_entity", context.get("likely_entity"))),
            "target_concept": text_value(target_record.get("concept_description"), target),
            "feature_context": feature_groups or ["unknown"],
            "semantic_summary": description,
            "domain_confidence": confidence,
        }

    def _verify_selector_integrity(self) -> None:
        checks = ((self.selector_path, EXPECTED_SELECTOR_SHA256), (self.selector_metadata_path, EXPECTED_SELECTOR_METADATA_SHA256))
        try:
            valid = all(path.is_file() and _sha256(path) == expected for path, expected in checks)
        except OSError as exc:
            raise PredictionExecutionFailure("selector_integrity_failure", "Frozen selector integrity could not be verified.", "model_selection") from exc
        if not valid:
            raise PredictionExecutionFailure("selector_integrity_failure", "Frozen selector integrity verification failed.", "model_selection")

    @staticmethod
    def _eligibility_failure(reasons: list[str]) -> PredictionExecutionFailure:
        if "cleaned_dataset_missing" in reasons:
            return PredictionExecutionFailure("cleaned_artifact_missing", "The prepared dataset is no longer available.", "task_resolution")
        if "cleaned_dataset_hash_mismatch" in reasons:
            return PredictionExecutionFailure("cleaned_artifact_hash_mismatch", "Prepared dataset integrity verification failed.", "task_resolution")
        if "prepared_dataset_has_duplicate_columns" in reasons:
            return PredictionExecutionFailure("target_duplicated", "The prepared dataset no longer has unique column names.", "task_resolution")
        return PredictionExecutionFailure("prediction_no_longer_eligible", "The prepared dataset is no longer eligible for prediction.", "task_resolution")

    @staticmethod
    def _safe_attempts(attempts: Any) -> list[dict[str, Any]]:
        safe: list[dict[str, Any]] = []
        for attempt in attempts if isinstance(attempts, list) else []:
            item = _mapping(attempt)
            status = item.get("status")
            safe.append({
                "rank": item.get("rank"), "model_id": item.get("model_id"), "status": status,
                "failure_stage": item.get("failure_stage"),
                "error": None if status == "success" else "Candidate execution failed for a technical reason.",
                "train_runtime_seconds": item.get("train_runtime_seconds"),
                "inference_runtime_seconds": item.get("inference_runtime_seconds"),
            })
        return safe

    def _build_result(self, run_id: str, prediction_id: str, target: str, task_resolution: dict[str, Any], orchestration: Mapping[str, Any], worker_seconds: float) -> dict[str, Any]:
        execution = _mapping(orchestration.get("execution"))
        predictions = _mapping(orchestration.get("predictions"))
        split = _mapping(orchestration.get("split"))
        selector = _mapping(orchestration.get("selector"))
        dataset = _mapping(orchestration.get("dataset"))
        artifacts = _mapping(orchestration.get("artifacts"))
        safe_attempts = self._safe_attempts(execution.get("attempts"))
        y_true = predictions.get("y_true") or []
        y_pred = predictions.get("y_pred") or []
        decoded = _mapping(predictions.get("decoded_labels"))
        if decoded:
            y_true = decoded.get("y_true") or y_true
            y_pred = decoded.get("y_pred") or y_pred
        probabilities = predictions.get("probabilities")
        fieldnames = ["actual", "prediction"]
        if isinstance(probabilities, list) and probabilities:
            width = len(probabilities[0]) if isinstance(probabilities[0], list) else 0
            fieldnames.extend(f"probability_{index}" for index in range(width))
        rows: list[list[Any]] = []
        for index, (actual, predicted) in enumerate(zip(y_true, y_pred)):
            row = [actual, predicted]
            if isinstance(probabilities, list) and index < len(probabilities) and isinstance(probabilities[index], list):
                row.extend(probabilities[index])
            rows.append(row)
        self.store.write_prediction_rows(run_id, prediction_id, fieldnames, rows)
        train_seconds = sum(float(item.get("train_runtime_seconds") or 0.0) for item in safe_attempts)
        inference_seconds = sum(float(item.get("inference_runtime_seconds") or 0.0) for item in safe_attempts)
        result = {
            "schema_version": "1.0", "run_id": run_id, "prediction_id": prediction_id,
            "target_column": target, "task_type": task_resolution.get("task_type"),
            "task_resolution": task_resolution,
            "selection": {
                "selected_model": execution.get("selected_model"), "selected_rank": execution.get("selected_rank"),
                "fallback_used": bool(execution.get("fallback_used")), "requested_model": selector.get("requested_model"),
                "candidate_ranking": selector.get("ranking"), "attempts": safe_attempts,
            },
            "evaluation": orchestration.get("evaluation"),
            "prediction_summary": {
                "row_count": len(rows), "probabilities_available": isinstance(probabilities, list),
                "probability_class_order": predictions.get("probability_class_order"), "artifact": "predictions.csv",
            },
            "runtime": {"train_runtime_seconds": train_seconds, "inference_runtime_seconds": inference_seconds, "worker_runtime_seconds": float(worker_seconds)},
            "orchestrator_audit": {
                "dataset": {"dataset_name": dataset.get("dataset_name"), "row_count": dataset.get("row_count"), "feature_names": dataset.get("feature_names")},
                "split": {"split_seed": split.get("split_seed"), "test_size": split.get("test_size"), "stratified": split.get("stratified"), "train_row_count": len(split.get("train_indices") or []), "test_row_count": len(split.get("test_indices") or [])},
                "selector": {"representation": selector.get("representation"), "semantic_profile": selector.get("semantic_profile"), "target_info": selector.get("target_info")},
                "artifacts": {
                    "model_reference": artifacts.get("model_reference"),
                    "preprocessor_reference": artifacts.get("preprocessor_reference"),
                    "target_encoder_reference": artifacts.get("target_encoder_reference"),
                    "transformed_feature_names": artifacts.get("transformed_feature_names"),
                },
                "warnings": orchestration.get("warnings") or [],
            },
        }
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        return result

    def _execute(self, run_id: str, prediction_id: str) -> None:
        started = time.perf_counter()
        stage = "task_resolution"
        try:
            state = self.store.transition_if(
                run_id, prediction_id, expected={"queued"}, state="running",
                stage=stage, error=None, worker_instance_id=self.instance_id,
                started_at=_utc_now(),
            )
            if state is None:
                return
            target = state["target_column"]
            context = self.run_service.prediction_context(run_id)
            eligibility = context["eligibility"]
            if not eligibility["eligible"]:
                raise self._eligibility_failure(list(eligibility.get("reasons") or []))
            occurrences = context["target_columns"].count(target)
            if occurrences == 0:
                raise PredictionExecutionFailure("target_disappeared", "The requested target is no longer present.", stage)
            if occurrences != 1:
                raise PredictionExecutionFailure("target_duplicated", "The requested target is no longer unique.", stage)

            import pandas as pd

            try:
                dataframe = pd.read_csv(context["cleaned_path"], low_memory=False)
            except (OSError, pd.errors.ParserError) as exc:
                raise PredictionExecutionFailure("cleaned_artifact_unreadable", "The prepared dataset could not be loaded.", stage) from exc
            dataframe_occurrences = list(dataframe.columns).count(target)
            if dataframe_occurrences != 1:
                code = "target_disappeared" if dataframe_occurrences == 0 else "target_duplicated"
                raise PredictionExecutionFailure(code, "The requested target is not present exactly once.", stage)
            report = context["pipeline1_report"]
            resolution = self._resolution_dict(self.resolver(dataframe, target, report))
            if resolution.get("status") != "resolved" or resolution.get("task_type") not in {"classification", "regression"}:
                self.store.transition_if(
                    run_id, prediction_id, expected={"running"}, owner=self.instance_id,
                    stage=stage, task_resolution=resolution,
                )
                raise PredictionExecutionFailure("task_type_unresolved", "Task type could not be resolved safely from the prepared dataset.", stage)

            stage = "model_selection"
            if self.store.transition_if(
                run_id, prediction_id, expected={"running"}, owner=self.instance_id,
                stage=stage, task_resolution=resolution,
            ) is None:
                return
            self._verify_selector_integrity()
            run_state = self.run_service.store.read(run_id)
            outcome = self.orchestrator(
                dataframe, target_column=target, task_type=resolution["task_type"],
                dataset_name=run_state["original_filename"], dataset_id=run_id,
                semantic_profile=self._semantic_profile(report, target), generate_semantic_profile=False,
                selector_path=self.selector_path, verify_selector_hash=True,
            )
            converter = getattr(outcome, "to_dict", None)
            orchestration = converter() if callable(converter) else outcome
            if not isinstance(orchestration, Mapping):
                raise PredictionExecutionFailure("selector_execution_failure", "Prediction orchestration returned an invalid result.", stage)
            execution = _mapping(orchestration.get("execution"))
            if execution.get("status") != "success":
                attempts = execution.get("attempts") or []
                code = "top2_fallback_technical_failure" if len(attempts) >= 2 else "top1_model_technical_failure"
                failure_stage = _mapping(attempts[-1]).get("failure_stage") if attempts else None
                terminal_stage = (
                    "evaluation" if failure_stage in {
                        "inference_preprocessing", "prediction", "evaluation"
                    } else "training"
                )
                raise PredictionExecutionFailure(
                    code, "No selected model could complete prediction.", terminal_stage
                )

            stage = "evaluation"
            if self.store.transition_if(
                run_id, prediction_id, expected={"running"}, owner=self.instance_id,
                stage=stage,
            ) is None:
                return
            try:
                result = self._build_result(run_id, prediction_id, target, resolution, orchestration, time.perf_counter() - started)
                self.store.write_result(run_id, prediction_id, result)
            except PredictionExecutionFailure:
                raise
            except Exception as exc:
                raise PredictionExecutionFailure("result_serialization_failure", "Prediction result could not be persisted safely.", stage) from exc
            self.store.transition_if(
                run_id, prediction_id, expected={"running"}, owner=self.instance_id,
                state="complete", stage="complete", error=None, completed_at=_utc_now(),
            )
        except PredictionExecutionFailure as exc:
            self.store.transition_if(
                run_id, prediction_id, expected={"running"}, owner=self.instance_id,
                state="failed", stage=exc.stage, error={"code": exc.code, "message": exc.message},
                completed_at=_utc_now(),
            )
        except ServiceError:
            logger.exception("Controlled prediction service failure for %s", prediction_id)
            self.store.transition_if(
                run_id, prediction_id, expected={"running"}, owner=self.instance_id,
                state="failed", stage=stage, error={"code": "prediction_no_longer_eligible", "message": "Prediction prerequisites are no longer available."},
                completed_at=_utc_now(),
            )
        except Exception:
            logger.exception("Prediction execution failed for %s", prediction_id)
            self.store.transition_if(
                run_id, prediction_id, expected={"running"}, owner=self.instance_id,
                state="failed", stage=stage, error={"code": "selector_execution_failure", "message": "Prediction execution could not complete."},
                completed_at=_utc_now(),
            )
        finally:
            with self._schedule_lock:
                self._created_predictions.discard((run_id, prediction_id))

    def status(self, run_id: str, prediction_id: str) -> dict[str, Any]:
        return self.store.read(run_id, prediction_id)

    def _completed_state(self, run_id: str, prediction_id: str) -> dict[str, Any]:
        state = self.store.read(run_id, prediction_id)
        if state["state"] in {"queued", "running"}:
            raise ServiceError(409, "prediction_not_complete", "Prediction result is not available yet.")
        if state["state"] == "failed":
            raise ServiceError(409, "prediction_failed", "Prediction did not complete successfully.")
        if state["state"] != "complete":
            raise ServiceError(500, "prediction_state_invalid", "Prediction state is invalid.")
        return state

    @staticmethod
    def _pagination_value(value: Any, *, name: str, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, bool):
            valid = False
        elif isinstance(value, int):
            valid = True
        elif isinstance(value, str):
            valid = value.isdecimal()
            if valid:
                value = int(value)
        else:
            valid = False
        if not valid:
            raise ServiceError(
                422, "invalid_pagination",
                "Prediction row pagination parameters are invalid.",
            )
        return int(value)

    def result(self, run_id: str, prediction_id: str) -> dict[str, Any]:
        self._completed_state(run_id, prediction_id)
        projected = project_frontend_result(self.store.read_result(run_id, prediction_id))
        rows_path = self.store._paths(run_id, prediction_id)["predictions"]
        if not rows_path.is_file() or rows_path.is_symlink():
            raise ServiceError(500, "prediction_rows_unavailable", "Prediction rows are unavailable.")
        return projected

    def _public_rows_source(self, run_id: str, prediction_id: str) -> tuple[str, Path]:
        self._completed_state(run_id, prediction_id)
        internal = self.store.read_result(run_id, prediction_id)
        task_type = internal.get("task_type")
        if task_type not in {"classification", "regression"}:
            raise ServiceError(500, "prediction_result_invalid", "Prediction result is invalid.")
        path = self.store._paths(run_id, prediction_id)["predictions"]
        if not path.is_file() or path.is_symlink():
            raise ServiceError(500, "prediction_rows_unavailable", "Prediction rows are unavailable.")
        return task_type, path

    @staticmethod
    def _validated_csv_rows(path: Path, task_type: str) -> Iterator[list[str]]:
        """Validate canonical CSV structure incrementally, without retaining rows."""
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, strict=True)
                try:
                    header = next(reader)
                except StopIteration as exc:
                    raise ServiceError(500, "prediction_rows_invalid", "Prediction rows are invalid.") from exc
                if len(header) != len(set(header)):
                    raise ServiceError(500, "prediction_rows_invalid", "Prediction rows are invalid.")
                if task_type == "classification":
                    valid_header = (
                        header[:2] == ["actual", "prediction"]
                        and all(name == f"probability_{index}" for index, name in enumerate(header[2:]))
                    )
                else:
                    valid_header = header == ["actual", "prediction"]
                if not valid_header:
                    raise ServiceError(500, "prediction_rows_invalid", "Prediction rows have an unexpected schema.")
                for row in reader:
                    if len(row) != len(header):
                        raise ServiceError(500, "prediction_rows_invalid", "Prediction rows are invalid.")
                    yield row
        except ServiceError:
            raise
        except FileNotFoundError as exc:
            raise ServiceError(500, "prediction_rows_unavailable", "Prediction rows are unavailable.") from exc
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ServiceError(500, "prediction_rows_invalid", "Prediction rows are invalid.") from exc

    def rows(
        self, run_id: str, prediction_id: str, *,
        offset: Any = 0, limit: Any = DEFAULT_PREDICTION_ROW_LIMIT,
    ) -> dict[str, Any]:
        """Read a bounded, task-aware projection of the canonical CSV rows."""
        self._completed_state(run_id, prediction_id)
        offset_value = self._pagination_value(offset, name="offset", default=0)
        limit_value = self._pagination_value(
            limit, name="limit", default=DEFAULT_PREDICTION_ROW_LIMIT
        )
        if offset_value < 0 or limit_value < 1 or limit_value > MAX_PREDICTION_ROW_LIMIT:
            raise ServiceError(
                422, "invalid_pagination",
                "Prediction row pagination parameters are invalid.",
            )

        task_type, path = self._public_rows_source(run_id, prediction_id)
        selected_rows: list[list[str]] = []
        total = 0
        for row in self._validated_csv_rows(path, task_type):
            if offset_value <= total < offset_value + limit_value:
                selected_rows.append(row)
            total += 1

        projected = [_public_prediction_row(row, task_type) for row in selected_rows]
        return {
            "prediction_id": prediction_id,
            "offset": offset_value,
            "limit": limit_value,
            "returned": len(projected),
            "total": total,
            "rows": projected,
        }

    def export(self, run_id: str, prediction_id: str) -> Iterator[str]:
        """Preflight malformed artifacts, then stream the complete safe CSV."""
        task_type, path = self._public_rows_source(run_id, prediction_id)
        # Streaming starts after HTTP headers are sent. Validate the whole file
        # first so malformed persisted rows return a controlled API error.
        for _row in self._validated_csv_rows(path, task_type):
            pass
        fields = (
            ("actual", "prediction", "confidence")
            if task_type == "classification" else
            ("actual", "prediction", "absolute_error")
        )

        def chunks() -> Iterator[str]:
            buffer = StringIO()
            writer = csv.writer(buffer, lineterminator="\r\n")
            writer.writerow(fields)
            yield buffer.getvalue()
            for row in self._validated_csv_rows(path, task_type):
                buffer.seek(0)
                buffer.truncate(0)
                public = _public_prediction_row(row, task_type)
                writer.writerow([_public_csv_cell(public[field]) for field in fields])
                yield buffer.getvalue()

        return chunks()

    def shutdown(self, *, wait: bool = True) -> None:
        with self._schedule_lock:
            self._shutting_down = True
        # Never hold the scheduling lock while waiting for callbacks that
        # remove completed work from the active-ID registry.
        if self.executor is not None:
            self.executor.shutdown(wait=wait, cancel_futures=not wait)


__all__ = [
    "MAX_PREDICTION_ROW_LIMIT", "PredictionExecutionFailure",
    "PredictionJobService", "PredictionStore", "project_frontend_result",
]
