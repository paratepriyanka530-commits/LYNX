"""Run the frozen selector and execute only its Top-1 recommendation.

Top-2 is attempted once, and only when Top-1 cannot execute for a technical
reason.  This module deliberately does not benchmark or compare candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import hashlib
import importlib
from pathlib import Path
import sys
import time
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from benchmarks.tabarena.evaluation_metrics import (
    classification_results,
    obtain_probabilities,
    regression_results,
    validate_probabilities,
)
from benchmarks.tabarena.pipeline2_groundwork import GroundworkOllamaProvider
from pipelines.pipeline2_prediction.dataset_semantic_profiler import (
    Pipeline2DatasetSemanticProfile,
    build_dataset_semantic_evidence,
    build_dataset_semantic_prompt,
)
from pipelines.pipeline2_prediction.model_selector.train_final_selector import (
    LYNXModelSelector,
)


ROOT = Path(__file__).resolve().parents[3]
SELECTOR_PATH = (
    ROOT / "pipelines" / "pipeline2_prediction" / "model_selector"
    / "lynx_model_selector.joblib"
)
FROZEN_SELECTOR_SHA256 = (
    "46ff89a509316389aa23b0f4b8083f2f83e8804759a4966e83e4616d59e06550"
)
CANONICAL_MODEL_IDS = (
    "random_forest", "xgboost", "lightgbm", "catboost", "xtab",
    "tabpfn", "tabm", "tabiclv2", "realmlp", "ft_transformer",
)
_EXTERNAL_PREPROCESSORS = {
    "xgboost": (
        "pipelines.pipeline2_prediction.models.xgboost.xgboost_preprocessor",
        "XGBoostPreprocessor",
    ),
    "lightgbm": (
        "pipelines.pipeline2_prediction.models.lightgbm.lightgbm_preprocessor",
        "LightGBMPreprocessor",
    ),
    "catboost": (
        "pipelines.pipeline2_prediction.models.catboost.catboost_preprocessor",
        "CatBoostPreprocessor",
    ),
}
_MAX_RETAINED_TRANSFORMED_CELLS = 1_000_000
_PERMUTATION_MAX_FEATURES = 25
_PERMUTATION_MAX_ROWS = 500
_PERMUTATION_REPEATS = 2


def _held_out_permutation_importance(
    *, package: Any, model: Any, preprocessor: Any, task_type: str,
    X_test: pd.DataFrame, y_test: np.ndarray, baseline_predictions: np.ndarray,
    random_state: int,
) -> dict[str, Any]:
    """Measure source-column importance on held-out rows without target leakage."""
    feature_names = [str(column) for column in X_test.columns]
    if not feature_names:
        return {"available": False, "reason": "No explanatory features are available."}
    if len(feature_names) > _PERMUTATION_MAX_FEATURES:
        return {
            "available": False,
            "reason": (
                f"Permutation importance is limited to {_PERMUTATION_MAX_FEATURES} "
                "source features for interactive prediction runs."
            ),
        }

    sample_size = min(len(X_test), _PERMUTATION_MAX_ROWS)
    if sample_size < 2:
        return {"available": False, "reason": "Too few held-out rows are available."}
    rng = np.random.default_rng(random_state)
    positions = (
        np.arange(len(X_test)) if sample_size == len(X_test)
        else np.sort(rng.choice(len(X_test), size=sample_size, replace=False))
    )
    X_sample = X_test.iloc[positions].copy()
    y_sample = np.asarray(y_test)[positions]
    baseline_sample = np.asarray(baseline_predictions)[positions]

    def score(actual: np.ndarray, predicted: np.ndarray) -> float:
        if task_type == "classification":
            return float(balanced_accuracy_score(actual, predicted))
        return -float(mean_squared_error(actual, predicted) ** 0.5)

    baseline_score = score(y_sample, baseline_sample)
    records: list[dict[str, Any]] = []
    for feature_name in feature_names:
        decreases = []
        original_values = X_sample[feature_name].to_numpy(copy=True)
        for _repeat in range(_PERMUTATION_REPEATS):
            permuted = X_sample.copy()
            permuted[feature_name] = rng.permutation(original_values)
            transformed = preprocessor.transform(permuted) if preprocessor is not None else permuted
            permuted_predictions = _validate_predictions(
                task_type, package.predict_model(model, transformed), y_sample
            )
            decreases.append(baseline_score - score(y_sample, permuted_predictions))
        records.append({"feature": feature_name, "importance": float(np.mean(decreases))})
    records.sort(key=lambda item: item["importance"], reverse=True)
    return {
        "available": True,
        "method": "held_out_permutation_importance",
        "metric": "balanced_accuracy_decrease" if task_type == "classification" else "rmse_increase",
        "interpretation": (
            "Mean held-out balanced-accuracy decrease after source-column permutation."
            if task_type == "classification" else
            "Mean held-out RMSE increase after source-column permutation, in target units."
        ),
        "evaluation_scope": "held_out_test_split",
        "repeats": _PERMUTATION_REPEATS,
        "sample_rows": sample_size,
        "random_state": int(random_state),
        "features": records[:10],
    }


@dataclass
class PredictionRuntimeHandles:
    """Non-serializable fitted state retained for downstream XAI/visualization."""

    fitted_model: Any = None
    fitted_preprocessor: Any = None
    fitted_target_encoder: LabelEncoder | None = None
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: Any = None
    y_test: Any = None
    X_train_transformed: Any = None
    X_test_transformed: Any = None
    transformed_feature_names: list[str] | None = None


@dataclass
class PredictionOrchestrationResult:
    """Serializable audit payload plus in-process fitted runtime objects."""

    result: dict[str, Any]
    runtime: PredictionRuntimeHandles = field(default_factory=PredictionRuntimeHandles)

    def to_dict(self) -> dict[str, Any]:
        return self.result


class CandidateExecutionError(RuntimeError):
    """Technical candidate failure annotated with its execution stage."""

    def __init__(
        self,
        stage: str,
        error: BaseException,
        train_runtime_seconds: float | None = None,
        inference_runtime_seconds: float | None = None,
    ):
        super().__init__(f"{type(error).__name__}: {error}")
        self.stage = stage
        self.original_error = error
        self.train_runtime_seconds = train_runtime_seconds
        self.inference_runtime_seconds = inference_runtime_seconds


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, Path)):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _json_list(values: Any) -> list[Any]:
    return [_json_value(value) for value in np.asarray(values).reshape(-1).tolist()]


def _selector_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_selector(
    selector_path: Path, verify_hash: bool,
) -> tuple[LYNXModelSelector, str]:
    path = Path(selector_path)
    if not path.is_file():
        raise FileNotFoundError(f"Frozen selector artifact not found: {path}")
    digest = _selector_digest(path)
    if verify_hash and digest != FROZEN_SELECTOR_SHA256:
        raise RuntimeError(
            "Frozen selector SHA-256 mismatch: "
            f"expected {FROZEN_SELECTOR_SHA256}, got {digest}"
        )
    return LYNXModelSelector.load(path), digest


def _validate_input(
    frame: pd.DataFrame, target_column: str, task_type: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("prepared_data must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("prepared_data cannot be empty")
    if not isinstance(target_column, str) or not target_column:
        raise ValueError("target_column must be a non-empty string")
    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"Duplicate column names are not allowed: {duplicate_columns}")
    if list(frame.columns).count(target_column) != 1:
        raise ValueError(f"Target column must exist exactly once: {target_column!r}")
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be exactly 'classification' or 'regression'")

    # The target is excluded before any model or semantic evidence sees X.
    X = frame.drop(columns=[target_column]).copy()
    y = frame[target_column].copy()
    if X.shape[1] == 0:
        raise ValueError("At least one predictor feature is required")
    return X, y, list(X.columns)


def _can_stratify(y: pd.Series, test_size: float | int) -> bool:
    counts = y.value_counts(dropna=False)
    if len(counts) < 2 or int(counts.min()) < 2:
        return False
    n_rows = len(y)
    n_test = int(np.ceil(n_rows * test_size)) if isinstance(test_size, float) else int(test_size)
    n_train = n_rows - n_test
    return n_test >= len(counts) and n_train >= len(counts)


def _split_data(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    split_seed: int,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, bool, list[str]]:
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1")
    valid_target = y.notna()
    dropped = int((~valid_target).sum())
    warnings = []
    if dropped:
        warnings.append(f"Dropped {dropped} rows with missing target values before splitting.")
    X_valid = X.loc[valid_target]
    y_valid = y.loc[valid_target]
    if len(y_valid) < 2:
        raise ValueError("At least two rows with non-missing targets are required")
    if task_type == "classification" and y_valid.nunique(dropna=True) < 2:
        raise ValueError("Classification target must contain at least two classes")
    if task_type == "regression":
        y_valid = pd.to_numeric(y_valid, errors="raise")

    stratified = task_type == "classification" and _can_stratify(y_valid, test_size)
    if task_type == "classification" and not stratified:
        warnings.append(
            "Classification split was not stratified because class counts or split "
            "sizes made stratification mathematically invalid."
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X_valid,
        y_valid,
        test_size=test_size,
        random_state=split_seed,
        stratify=y_valid if stratified else None,
    )
    return X_train, X_test, y_train, y_test, stratified, warnings


def _encode_target(
    y_train_raw: pd.Series, y_test_raw: pd.Series, task_type: str,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder | None, dict[str, Any]]:
    if task_type == "regression":
        y_train = pd.to_numeric(y_train_raw, errors="raise").to_numpy(dtype=float)
        y_test = pd.to_numeric(y_test_raw, errors="raise").to_numpy(dtype=float)
        target_info = {
            "classification_type": None,
            "n_classes": None,
            "minority_class_ratio": None,
            "target_missing_pct": 0.0,
            "cb_chl_eligible": False,
        }
        return y_train, y_test, None, target_info

    encoder = LabelEncoder()
    y_train_text = y_train_raw.astype(str)
    y_train = encoder.fit_transform(y_train_text)
    try:
        y_test = encoder.transform(y_test_raw.astype(str))
    except ValueError as exc:
        raise ValueError(
            "The test split contains a target class absent from the training split"
        ) from exc
    counts = pd.Series(y_train).value_counts(normalize=True)
    n_classes = int(len(encoder.classes_))
    minority_ratio = float(counts.min())
    target_info = {
        "classification_type": "binary" if n_classes == 2 else "multiclass",
        "n_classes": n_classes,
        "minority_class_ratio": minority_ratio,
        # Missing targets were excluded before the supervised split; this value
        # describes the actual training target supplied to selector/model fitting.
        "target_missing_pct": 0.0,
        "cb_chl_eligible": minority_ratio < 0.01,
    }
    return y_train, y_test, encoder, target_info


def _feature_type_counts(X: pd.DataFrame) -> dict[str, int]:
    numeric = len(X.select_dtypes(include="number").columns)
    boolean = len(X.select_dtypes(include="bool").columns)
    datetime_count = len(X.select_dtypes(include=["datetime", "datetimetz"]).columns)
    categorical = len(X.select_dtypes(include=["object", "string", "category"]).columns)
    other = len(X.columns) - len(set(
        X.select_dtypes(include=[
            "number", "bool", "datetime", "datetimetz", "object", "string", "category",
        ]).columns
    ))
    return {
        "numeric": numeric,
        "categorical": categorical,
        "boolean": boolean,
        "datetime": datetime_count,
        "other": max(0, other),
    }


def _semantic_profile(
    *,
    supplied: Mapping[str, Any] | Pipeline2DatasetSemanticProfile | None,
    generate: bool,
    dataset_name: str,
    task_type: str,
    target_column: str,
    X: pd.DataFrame,
    model: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    if supplied is not None:
        return Pipeline2DatasetSemanticProfile.model_validate(supplied).model_dump(mode="json")
    if not generate:
        raise ValueError(
            "A valid semantic_profile is required unless generate_semantic_profile=True"
        )
    evidence = build_dataset_semantic_evidence(
        dataset_name=dataset_name,
        task_type=task_type,
        target_column=target_column,
        row_count=len(X),
        feature_count=X.shape[1],
        feature_type_counts=_feature_type_counts(X),
        feature_names=list(X.columns),
    )
    prompt = build_dataset_semantic_prompt(evidence)
    provider = GroundworkOllamaProvider(model=model, timeout_seconds=timeout_seconds)
    response = provider.generate_structured(prompt, Pipeline2DatasetSemanticProfile)
    return Pipeline2DatasetSemanticProfile.model_validate(response).model_dump(mode="json")


def _build_external_preprocessor(model_id: str) -> Any:
    specification = _EXTERNAL_PREPROCESSORS.get(model_id)
    if specification is None:
        return None
    module_name, class_name = specification
    return getattr(importlib.import_module(module_name), class_name)()


def _import_model_package(model_id: str) -> Any:
    return importlib.import_module(
        f"pipelines.pipeline2_prediction.models.{model_id}"
    )


def _transformed_feature_names(model_id: str, model: Any, transformed: Any) -> list[str] | None:
    if isinstance(transformed, pd.DataFrame):
        return [str(column) for column in transformed.columns]
    if model_id == "random_forest":
        try:
            values = model.named_steps["preprocessor"].get_feature_names_out()
            return [str(value) for value in values]
        except (AttributeError, KeyError, RuntimeError, ValueError):
            return None
    return None


def _validate_predictions(task_type: str, predictions: Any, y_test: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions).reshape(-1)
    if len(values) != len(y_test):
        raise ValueError(
            f"Prediction length {len(values)} does not match test rows {len(y_test)}"
        )
    if task_type == "regression":
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Regression predictions contain NaN or infinite values")
    else:
        try:
            values = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(
                dtype=np.asarray(y_test).dtype
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Classification predictions are not valid encoded labels") from exc
        allowed = set(np.unique(y_test).tolist())
        # Models may validly predict a training class absent from y_test, so the
        # caller validates against the complete fitted encoder separately.
        if not np.isfinite(values.astype(float)).all():
            raise ValueError("Classification predictions contain non-finite values")
    return values


def _probability_payload(
    model_id: str, model: Any, X_test: Any, y_train: np.ndarray, y_test: np.ndarray,
) -> tuple[list[list[float]] | None, list[Any] | None, str | None]:
    expected_classes = np.unique(y_train)
    probabilities, classes, status, reason = obtain_probabilities(
        model_id, model, X_test, expected_classes
    )
    if status != "available":
        return None, None, reason
    probabilities, classes, invalid_reason = validate_probabilities(
        probabilities, classes, y_test, expected_classes
    )
    if invalid_reason:
        return None, None, invalid_reason
    return probabilities.tolist(), _json_list(classes), None


def _release_failed_candidate(model: Any) -> None:
    del model
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _execute_candidate(
    *,
    ranking_record: dict[str, Any],
    task_type: str,
    split_seed: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_encoder: LabelEncoder | None,
) -> tuple[dict[str, Any], dict[str, Any], PredictionRuntimeHandles]:
    model_id = str(ranking_record["model_name"])
    rank = int(ranking_record["rank"])
    stage = "import"
    model = None
    preprocessor = None
    train_seconds = None
    inference_seconds = None
    inference_started = None
    train_started = time.perf_counter()
    try:
        package = _import_model_package(model_id)
        stage = "preprocessing"
        preprocessor = _build_external_preprocessor(model_id)
        X_train_model = (
            preprocessor.fit_transform(X_train)
            if preprocessor is not None else X_train.copy()
        )
        stage = "model_build"
        model = package.build_model(task_type, random_state=split_seed)
        stage = "model_fit"
        package.train_model(model, X_train_model, y_train)
        train_seconds = float(time.perf_counter() - train_started)

        stage = "inference_preprocessing"
        inference_started = time.perf_counter()
        X_test_model = (
            preprocessor.transform(X_test)
            if preprocessor is not None else X_test.copy()
        )
        stage = "prediction"
        predictions = _validate_predictions(
            task_type, package.predict_model(model, X_test_model), y_test
        )
        stage = "evaluation"
        probability_values = None
        probability_classes = None
        probability_warning = None
        if task_type == "classification":
            predictive, calibration, probability_metadata = classification_results(
                y_train, y_test, predictions, model_id, model, X_test_model
            )
            probability_values, probability_classes, probability_warning = _probability_payload(
                model_id, model, X_test_model, y_train, y_test
            )
        else:
            predictive, calibration, probability_metadata = regression_results(
                y_test, predictions
            )
        try:
            feature_importance = _held_out_permutation_importance(
                package=package, model=model, preprocessor=preprocessor,
                task_type=task_type, X_test=X_test, y_test=y_test,
                baseline_predictions=predictions, random_state=split_seed,
            )
        except Exception:
            feature_importance = {
                "available": False,
                "reason": "Reliable held-out permutation importance could not be computed for this model.",
            }
        inference_seconds = float(time.perf_counter() - inference_started)

        transformed_names = _transformed_feature_names(model_id, model, X_train_model)
        decoded = None
        if target_encoder is not None:
            decoded = {
                "y_true": target_encoder.inverse_transform(y_test).tolist(),
                "y_pred": target_encoder.inverse_transform(predictions.astype(int)).tolist(),
                "class_order": target_encoder.classes_.tolist(),
            }
        attempt = {
            "rank": rank,
            "model_id": model_id,
            "status": "success",
            "failure_stage": None,
            "error": None,
            "train_runtime_seconds": train_seconds,
            "inference_runtime_seconds": inference_seconds,
        }
        serializable = {
            "attempt": attempt,
            "predictions": {
                "y_true": _json_list(y_test),
                "y_pred": _json_list(predictions),
                "probabilities": probability_values,
                "probability_class_order": probability_classes,
                "decoded_labels": decoded,
            },
            "evaluation": {
                "predictive_metrics": predictive,
                "calibration_metrics": calibration,
                "probability_metadata": probability_metadata,
                "feature_importance": feature_importance,
            },
            "artifacts": {
                "model_artifact": None,
                "model_reference": f"{type(model).__module__}.{type(model).__name__}",
                "preprocessor_artifact": None,
                "preprocessor_reference": (
                    None if preprocessor is None
                    else f"{type(preprocessor).__module__}.{type(preprocessor).__name__}"
                ),
                "target_encoder_artifact": None,
                "target_encoder_reference": (
                    None if target_encoder is None
                    else f"{type(target_encoder).__module__}.{type(target_encoder).__name__}"
                ),
                "transformed_feature_names": transformed_names,
            },
            "warnings": ([f"Probability output unavailable: {probability_warning}"]
                         if probability_warning else []),
        }
        retain_transformed = (
            getattr(X_train_model, "shape", (0, 0))[0]
            * getattr(X_train_model, "shape", (0, 0))[1]
            <= _MAX_RETAINED_TRANSFORMED_CELLS
        )
        runtime = PredictionRuntimeHandles(
            fitted_model=model,
            fitted_preprocessor=preprocessor,
            fitted_target_encoder=target_encoder,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_train_transformed=X_train_model if retain_transformed else None,
            X_test_transformed=X_test_model if retain_transformed else None,
            transformed_feature_names=transformed_names,
        )
        return attempt, serializable, runtime
    except Exception as exc:
        if train_seconds is None:
            train_seconds = float(time.perf_counter() - train_started)
        elif inference_started is not None:
            inference_seconds = float(time.perf_counter() - inference_started)
        if model is not None:
            _release_failed_candidate(model)
        raise CandidateExecutionError(
            stage,
            exc,
            train_runtime_seconds=train_seconds,
            inference_runtime_seconds=inference_seconds,
        ) from exc


def _validate_ranking(ranking: list[dict[str, Any]]) -> None:
    model_ids = [str(record.get("model_name")) for record in ranking]
    ranks = [record.get("rank") for record in ranking]
    if len(ranking) != len(CANONICAL_MODEL_IDS):
        raise RuntimeError("Frozen selector did not return all ten canonical candidates")
    if set(model_ids) != set(CANONICAL_MODEL_IDS) or len(model_ids) != len(set(model_ids)):
        raise RuntimeError("Frozen selector returned an invalid canonical model set")
    if ranks != list(range(1, len(CANONICAL_MODEL_IDS) + 1)):
        raise RuntimeError("Frozen selector ranking is not consecutively ordered")


def orchestrate_prediction(
    prepared_data: pd.DataFrame,
    *,
    target_column: str,
    task_type: str,
    dataset_name: str,
    dataset_id: str | int | None = None,
    semantic_profile: Mapping[str, Any] | Pipeline2DatasetSemanticProfile | None = None,
    generate_semantic_profile: bool = False,
    semantic_model: str = "qwen2.5:7b",
    semantic_timeout_seconds: float = 600.0,
    split_seed: int = 42,
    test_size: float = 0.2,
    selector_path: str | Path = SELECTOR_PATH,
    verify_selector_hash: bool = True,
) -> PredictionOrchestrationResult:
    """Select and execute one model, with at most one technical fallback."""
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string")
    X, y, feature_names = _validate_input(prepared_data, target_column, task_type)
    X_train, X_test, y_train_raw, y_test_raw, stratified, warnings = _split_data(
        X, y, task_type, split_seed, test_size
    )
    y_train, y_test, target_encoder, target_info = _encode_target(
        y_train_raw, y_test_raw, task_type
    )
    semantics = _semantic_profile(
        supplied=semantic_profile,
        generate=generate_semantic_profile,
        dataset_name=dataset_name,
        task_type=task_type,
        target_column=target_column,
        X=X,
        model=semantic_model,
        timeout_seconds=semantic_timeout_seconds,
    )
    selector, selector_hash = _load_frozen_selector(
        Path(selector_path), verify_selector_hash
    )
    ranking = selector.predict_ranking(
        task_type=task_type,
        target_info=target_info,
        semantic_features=semantics,
    )
    _validate_ranking(ranking)

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-" + uuid4().hex[:8]
    )
    attempts: list[dict[str, Any]] = []
    selected_payload = None
    runtime = PredictionRuntimeHandles(
        fitted_target_encoder=target_encoder,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    errors = []
    for candidate in ranking[:2]:
        try:
            attempt, selected_payload, runtime = _execute_candidate(
                ranking_record=candidate,
                task_type=task_type,
                split_seed=split_seed,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                target_encoder=target_encoder,
            )
            attempts.append(attempt)
            warnings.extend(selected_payload.pop("warnings"))
            break
        except CandidateExecutionError as exc:
            attempt = {
                "rank": int(candidate["rank"]),
                "model_id": str(candidate["model_name"]),
                "status": "failed",
                "failure_stage": exc.stage,
                "error": str(exc),
                "train_runtime_seconds": exc.train_runtime_seconds,
                "inference_runtime_seconds": exc.inference_runtime_seconds,
            }
            attempts.append(attempt)
            errors.append(
                f"rank_{candidate['rank']}:{candidate['model_name']}:{exc.stage}:{exc}"
            )

    success = selected_payload is not None
    selected_attempt = attempts[-1] if success else None
    result = {
        "schema_version": "1.0",
        "run_id": run_id,
        "dataset": {
            "dataset_id": None if dataset_id is None else str(dataset_id),
            "dataset_name": dataset_name,
            "target_column": target_column,
            "task_type": task_type,
            "row_count": int(len(prepared_data)),
            "feature_names": feature_names,
        },
        "split": {
            "split_seed": int(split_seed),
            "test_size": float(test_size),
            "train_indices": [_json_value(value) for value in X_train.index.tolist()],
            "test_indices": [_json_value(value) for value in X_test.index.tolist()],
            "stratified": bool(stratified),
        },
        "selector": {
            "artifact_path": str(Path(selector_path).resolve()),
            "artifact_sha256": selector_hash,
            "representation": selector.representation,
            "semantic_profile": semantics,
            "target_info": target_info,
            "ranking": ranking,
            "requested_model": str(ranking[0]["model_name"]),
        },
        "execution": {
            "status": "success" if success else "failed",
            "selected_model": selected_attempt["model_id"] if selected_attempt else None,
            "selected_rank": selected_attempt["rank"] if selected_attempt else None,
            "fallback_used": bool(success and selected_attempt["rank"] == 2),
            "attempts": attempts,
        },
        "predictions": selected_payload["predictions"] if success else None,
        "evaluation": selected_payload["evaluation"] if success else None,
        "artifacts": selected_payload["artifacts"] if success else {
            "model_artifact": None,
            "model_reference": None,
            "preprocessor_artifact": None,
            "preprocessor_reference": None,
            "target_encoder_artifact": None,
            "target_encoder_reference": None,
            "transformed_feature_names": None,
        },
        "warnings": warnings,
        "errors": errors,
    }
    return PredictionOrchestrationResult(result=result, runtime=runtime)


__all__ = [
    "FROZEN_SELECTOR_SHA256",
    "PredictionOrchestrationResult",
    "PredictionRuntimeHandles",
    "orchestrate_prediction",
]
