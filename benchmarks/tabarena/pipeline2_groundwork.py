"""Auditable data-generation groundwork for the Pipeline 2 model router.

This module deliberately does not train a router or invent missing scores.  It
builds a verified OpenML/TabArena manifest, computes deterministic statistical
meta-features, runs only Pipeline 1's semantic profiler, and validates rows
before they enter the model-run log.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASET_DIR = ROOT / "benchmarks" / "tabarena" / "datasets"
OUTPUT_DIR = ROOT / "benchmarks" / "tabarena" / "pipeline2_artifacts"
SUITE_ID = 457
FOUNDATION_MODEL_LIMITS = {
    "tabpfn": {
        "max_training_samples": 10_000,
        "max_features": 500,
        "max_classes": 10,
    },
}
MODEL_CAPABILITIES = {
    "randomforest": {
        "task_type_supported": ["classification", "regression"],
        "minimum_training_rows": None,
        "maximum_training_rows": None,
        "maximum_features": None,
        "maximum_classes": None,
        "binary_classification": True,
        "multiclass_classification": True,
        "missing_values": "No wrapper-level prohibition; categorical values use OneHotEncoder(handle_unknown='ignore'). Numeric missing-value artifact behavior is not separately certified.",
        "categorical_features": "Supported through the fitted OneHotEncoder.",
        "numeric_features": "Supported.",
        "artifact_requirements": "The supplied unit_price regression artifact requires its persisted 11-column pizza schema.",
        "environment_restrictions": "Supplied artifact verified with scikit-learn 1.9.0 and joblib 1.6.0.",
        "source": "random_forest package and fitted random_forest_metadata.pkl",
    },
    "lightgbm": {
        "task_type_supported": ["classification"],
        "minimum_training_rows": 2,
        "maximum_training_rows": None,
        "maximum_features": None,
        "maximum_classes": 2,
        "binary_classification": True,
        "multiclass_classification": False,
        "missing_values": "Fitted medians handle numeric nulls; categorical nulls become 'Unknown'.",
        "categorical_features": "Label-encoded; unseen inference labels are rejected by the fitted encoder.",
        "numeric_features": "Supported.",
        "artifact_requirements": "Supplied artifact requires the four persisted blood-donation feature names.",
        "environment_restrictions": "Verified in the pinned Pipeline 2 model environment.",
        "source": "lightgbm_config.py binary objective, model validation, and fitted preprocessor",
    },
    "xgboost": {
        "task_type_supported": ["classification"],
        "minimum_training_rows": 2,
        "maximum_training_rows": None,
        "maximum_features": None,
        "maximum_classes": 2,
        "binary_classification": True,
        "multiclass_classification": False,
        "missing_values": "Training preprocessing fills nulls; inference-time fill state is not persisted by the current preprocessor.",
        "categorical_features": "Label-encoded; unseen inference labels are rejected.",
        "numeric_features": "Supported.",
        "artifact_requirements": "Model and matching fitted preprocessor must be loaded together.",
        "environment_restrictions": "Pinned XGBoost 3.3.0 / scikit-learn 1.6.1 environment required for verified reuse.",
        "source": "xgboost_config.py binary:logistic objective, model validation, and preprocessor",
    },
    "catboost": {
        "task_type_supported": ["classification"],
        "minimum_training_rows": 1,
        "maximum_training_rows": None,
        "maximum_features": None,
        "maximum_classes": 2,
        "binary_classification": True,
        "multiclass_classification": False,
        "missing_values": "Fitted medians handle numeric nulls; categorical nulls become 'Unknown'.",
        "categorical_features": "Supported as object/string columns by the fitted preprocessor and CatBoost.",
        "numeric_features": "Supported for int64/float64 columns.",
        "artifact_requirements": "Supplied artifact requires the four persisted blood-donation feature names.",
        "environment_restrictions": "Verified in the pinned Pipeline 2 model environment.",
        "source": "catboost_config.py Logloss objective, model validation, and fitted preprocessor",
    },
    "tabpfn": {
        "task_type_supported": ["classification"],
        "minimum_training_rows": 2,
        "maximum_training_rows": 10_000,
        "maximum_features": 500,
        "maximum_classes": 10,
        "binary_classification": True,
        "multiclass_classification": True,
        "missing_values": "No repository preprocessing; values are passed unchanged to TabPFNClassifier.",
        "categorical_features": "No repository encoding or categorical-feature indices are supplied; compatibility is delegated to TabPFN.",
        "numeric_features": "Supported.",
        "artifact_requirements": "Pretrained classifier artifact and at least two target classes are required.",
        "environment_restrictions": "Verified with tabpfn 8.5.0, torch 2.11.0+cu128, and scikit-learn 1.6.1.",
        "source": "tabpfn_model.py plus TabPFN 8.5 inference_config pretraining limits",
    },
    "xtab": {
        "task_type_supported": ["classification", "regression"],
        "minimum_training_rows": 10,
        "maximum_training_rows": None,
        "maximum_features": None,
        "maximum_classes": None,
        "binary_classification": True,
        "multiclass_classification": True,
        "missing_values": "Numeric nulls use fitted means at inference; categorical nulls use a learned missing token. Regression targets cannot be missing/non-numeric.",
        "categorical_features": "Supported with learned mappings and an unknown-category index.",
        "numeric_features": "Supported with a fitted StandardScaler.",
        "artifact_requirements": "The official iter_2k checkpoint supplies 46 pretrained shared-backbone tensors before optimization. Dataset-specific tokenizer, normalization, head, and fitted preprocessor are created per dataset and persisted in the downstream model artifact.",
        "environment_restrictions": "Official backbone load and downstream train/predict verified with torch 2.11.0+cu128.",
        "source": "xtab_trainer.py, xtab_preprocessor.py, xtab_model.py, official XTab iter_2k checkpoint, and exact tensor-load verification",
    },
    "realmlp": {
        "task_type_supported": ["classification", "regression"],
        "minimum_training_rows": 2,
        "maximum_training_rows": None,
        "maximum_features": None,
        "maximum_classes": None,
        "binary_classification": True,
        "multiclass_classification": True,
        "missing_values": "LYNX fits numeric medians on train only; pytabkit handles categorical missing/unknown values through its fitted converter.",
        "categorical_features": "pytabkit RealMLP-TD internally one-hot encodes cardinality <=8 (threshold 9 includes unknown) and embeds larger cardinalities.",
        "numeric_features": "pytabkit internally applies median centering, robust scaling, smooth clipping, and numerical embeddings.",
        "artifact_requirements": "Fresh fitted PytabkitCandidate containing its train-fitted LYNX preprocessor and pytabkit estimator.",
        "environment_restrictions": "pytabkit 1.7.3; GPU optional, CPU supported.",
        "source": "pytabkit RealMLP_TD_Classifier/Regressor interfaces and installed source defaults",
    },
    "tabm": {
        "task_type_supported": ["classification", "regression"],
        "minimum_training_rows": 2,
        "maximum_training_rows": None,
        "maximum_features": None,
        "maximum_classes": None,
        "binary_classification": True,
        "multiclass_classification": True,
        "missing_values": "LYNX fits numeric medians on train only; pytabkit's fitted converter handles categorical missing/unknown values.",
        "categorical_features": "pytabkit TabM-D internally one-hot encodes categorical values.",
        "numeric_features": "pytabkit internally applies its configured quantile-style transformation.",
        "artifact_requirements": "Fresh fitted PytabkitCandidate containing its train-fitted LYNX preprocessor and pytabkit estimator.",
        "environment_restrictions": "pytabkit 1.7.3; GPU optional, CPU and MPS supported.",
        "source": "pytabkit TabM_D_Classifier/Regressor interfaces and installed source defaults",
    },
    "tabiclv2": {
        "task_type_supported": ["classification", "regression"],
        "minimum_training_rows": 2,
        "maximum_training_rows": None,
        "maximum_features": None,
        "maximum_classes": None,
        "binary_classification": True,
        "multiclass_classification": True,
        "missing_values": "LYNX fits train-only numeric medians and categorical missing tokens; TabICLv2 also natively handles numeric and categorical missing values.",
        "categorical_features": "Official tabicl preprocessing detects and ordinal-encodes pandas categorical/object/string/bool columns.",
        "numeric_features": "Supported with upstream scaling and normalization.",
        "artifact_requirements": "Official v2 classification or regression checkpoint, cached by tabicl.",
        "environment_restrictions": "tabicl 2.2.0; GPU optional but recommended for larger tables.",
        "source": "official soda-inria/tabicl 2.2.0 package and v2 checkpoints",
    },
    "fttransformer": {
        "task_type_supported": ["classification", "regression"],
        "minimum_training_rows": 2,
        "maximum_training_rows": None,
        "maximum_features": None,
        "maximum_classes": None,
        "binary_classification": True,
        "multiclass_classification": True,
        "missing_values": "LYNX fits train-only numeric medians and categorical missing tokens before pytabkit's fitted converter.",
        "categorical_features": "Supported through pytabkit's train-fitted FT-Transformer conversion.",
        "numeric_features": "Supported through pytabkit's configured quantile transformation.",
        "artifact_requirements": "Fresh fitted PytabkitCandidate; no pretrained checkpoint is loaded.",
        "environment_restrictions": "pytabkit 1.7.3 plus skorch 1.4.0; CPU, CUDA, and MPS supported.",
        "source": "pytabkit FTT_D_Classifier/Regressor (Gorishniy et al. FT-Transformer)",
    },
}
EVALUATION_MODEL_IDS = (
    "random_forest", "lightgbm", "xgboost", "catboost", "tabpfn", "xtab",
    "realmlp", "tabm", "tabiclv2", "ft_transformer",
)
LOG_COLUMNS = [
    "dataset_id", "model_id", "target_column", "task_type", "feasible",
    "status", "outcome_reason", "metric_name", "metric_value",
    "baseline_relative_score", "train_runtime_seconds",
    "inference_runtime_seconds", "runtime_seconds", "split_seed", "hyperparams",
    "semantic_meta_features", "statistical_meta_features", "source",
]
VALID_SOURCES = {
    "TabArena-published", "team-reproduced", "team-original-run",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _safe_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_") + ".csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _api_json(url: str, timeout: float) -> dict[str, Any]:
    import requests

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_manifest(timeout: float = 120) -> list[dict[str, Any]]:
    """Fetch the authoritative 51-task suite and reconcile it to local CSVs."""
    suite = _api_json(f"https://www.openml.org/api/v1/json/study/{SUITE_ID}", timeout)
    task_ids = suite["study"]["tasks"]["task_id"]
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = _api_json(f"https://www.openml.org/api/v1/json/task/{task_id}", timeout)["task"]
        source = next(item for item in task["input"] if item["name"] == "source_data")["data_set"]
        dataset_id = str(source["data_set_id"])
        description = _api_json(
            f"https://www.openml.org/api/v1/json/data/{dataset_id}", timeout
        )["data_set_description"]
        name = description["name"]
        filename = _safe_filename(name)
        local_path = DATASET_DIR / filename
        rows.append({
            "suite_id": SUITE_ID,
            "task_id": int(task_id),
            "openml_dataset_id": int(dataset_id),
            "dataset_name": name,
            "task_type": task["task_type"],
            "target_column": source.get("target_feature") or description.get("default_target_attribute"),
            "csv_file": filename,
            "local_csv_present": local_path.is_file(),
            "local_csv_bytes": local_path.stat().st_size if local_path.is_file() else None,
            "local_csv_sha256": _sha256(local_path) if local_path.is_file() else None,
            "openml_download_url": description.get("url"),
        })
    if len(rows) != 51:
        raise ValueError(f"OpenML suite {SUITE_ID} returned {len(rows)} tasks, expected 51")
    return rows


def sync_missing_datasets(manifest: list[dict[str, Any]], timeout: float = 240) -> list[dict[str, Any]]:
    """Download only absent official OpenML ARFF files and convert them to CSV."""
    import requests

    try:
        import arff
    except ImportError as exc:
        raise RuntimeError("sync requires the small 'liac-arff' package") from exc
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for item in manifest:
        destination = DATASET_DIR / item["csv_file"]
        if destination.is_file():
            continue
        temporary = destination.with_suffix(".arff.tmp")
        try:
            response = requests.get(item["openml_download_url"], timeout=timeout)
            response.raise_for_status()
            temporary.write_bytes(response.content)
            with temporary.open("r", encoding="utf-8", errors="replace") as handle:
                document = arff.load(handle)
            columns = [attribute[0] for attribute in document["attributes"]]
            frame = pd.DataFrame(document["data"], columns=columns)
            target = item["target_column"]
            if target not in frame.columns:
                raise ValueError(f"Official target column {target!r} is absent")
            frame.to_csv(destination, index=False)
            results.append({
                "dataset_id": str(item["task_id"]), "dataset_name": item["dataset_name"],
                "status": "downloaded", "rows": len(frame), "columns": len(frame.columns),
                "csv_file": item["csv_file"], "sha256": _sha256(destination),
            })
        except Exception as exc:
            results.append({
                "dataset_id": str(item["task_id"]), "dataset_name": item["dataset_name"],
                "status": "error", "error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            if temporary.is_file():
                temporary.unlink()
    return results


def write_manifest(rows: list[dict[str, Any]], output_dir: Path) -> None:
    _atomic_json(output_dir / "dataset_manifest.json", {
        "generated_at": _utc_now(), "source": f"OpenML suite {SUITE_ID}", "datasets": rows,
    })
    pd.DataFrame(rows).to_csv(output_dir / "dataset_manifest.csv", index=False)


def refresh_local_manifest_state(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for item in rows:
        path = DATASET_DIR / item["csv_file"]
        item["local_csv_present"] = path.is_file()
        item["local_csv_bytes"] = path.stat().st_size if path.is_file() else None
        item["local_csv_sha256"] = _sha256(path) if path.is_file() else None
    return rows


def _finite(values: Iterable[Any]) -> list[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _summary(values: Iterable[Any]) -> dict[str, float | None]:
    finite = _finite(values)
    if not finite:
        return {"mean": None, "median": None, "max_abs": None}
    series = pd.Series(finite, dtype="float64")
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "max_abs": float(series.abs().max()),
    }


def statistical_meta_features(dataframe: pd.DataFrame, target_column: str | None) -> dict[str, Any]:
    rows, columns = dataframe.shape
    feature_frame = dataframe.drop(columns=[target_column]) if target_column in dataframe.columns else dataframe
    numeric = feature_frame.select_dtypes(include="number")
    boolean = feature_frame.select_dtypes(include="bool")
    datetime_columns = feature_frame.select_dtypes(include=["datetime", "datetimetz"])
    categorical_count = max(0, feature_frame.shape[1] - numeric.shape[1] - boolean.shape[1] - datetime_columns.shape[1])
    denominator = max(1, feature_frame.shape[1])
    cardinalities = feature_frame.nunique(dropna=True)
    high_cardinality = (cardinalities / max(1, rows) >= 0.9).sum()
    target_present = bool(target_column and target_column in dataframe.columns)
    target = dataframe[target_column] if target_present else None
    target_counts = target.value_counts(dropna=True) if target is not None else pd.Series(dtype="int64")
    skew = numeric.skew(axis=0, skipna=True) if not numeric.empty else []
    kurtosis = numeric.kurt(axis=0, skipna=True) if not numeric.empty else []
    return {
        "row_count": int(rows),
        "column_count_including_target": int(columns),
        "feature_count": int(feature_frame.shape[1]),
        "missing_cell_ratio": float(dataframe.isna().to_numpy().mean()) if rows and columns else 0.0,
        "rows_with_missing_ratio": float(dataframe.isna().any(axis=1).mean()) if rows else 0.0,
        "duplicate_row_ratio": float(dataframe.duplicated().mean()) if rows else 0.0,
        "numeric_feature_ratio": float(numeric.shape[1] / denominator),
        "categorical_feature_ratio": float(categorical_count / denominator),
        "boolean_feature_ratio": float(boolean.shape[1] / denominator),
        "datetime_feature_ratio": float(datetime_columns.shape[1] / denominator),
        "high_cardinality_feature_ratio": float(high_cardinality / denominator),
        "mean_feature_cardinality": float(cardinalities.mean()) if len(cardinalities) else 0.0,
        "numeric_skew": _summary(skew),
        "numeric_kurtosis": _summary(kurtosis),
        "target_present": target_present,
        "target_missing_ratio": float(target.isna().mean()) if target is not None and rows else None,
        "target_class_count": int(len(target_counts)) if target is not None else None,
        "target_majority_ratio": float(target_counts.iloc[0] / target_counts.sum()) if len(target_counts) else None,
        "target_minority_to_majority_ratio": (
            float(target_counts.iloc[-1] / target_counts.iloc[0]) if len(target_counts) > 1 else None
        ),
    }


def compute_statistical_artifacts(manifest: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    records = []
    for item in manifest:
        path = DATASET_DIR / item["csv_file"]
        if not path.is_file():
            records.append({
                "dataset_id": str(item["task_id"]), "dataset_name": item["dataset_name"],
                "status": "missing_local_csv", "target_column": item["target_column"],
                "statistical_meta_features": None,
            })
            continue
        try:
            frame = pd.read_csv(path, low_memory=False)
            features = statistical_meta_features(frame, item["target_column"])
            status = "complete" if features["target_present"] else "target_column_missing"
            records.append({
                "dataset_id": str(item["task_id"]), "dataset_name": item["dataset_name"],
                "status": status, "target_column": item["target_column"],
                "statistical_meta_features": features,
            })
        except Exception as exc:
            records.append({
                "dataset_id": str(item["task_id"]), "dataset_name": item["dataset_name"],
                "status": "error", "target_column": item["target_column"],
                "statistical_meta_features": None, "error": f"{type(exc).__name__}: {exc}",
            })
    _atomic_json(output_dir / "statistical_meta_features.json", {
        "generated_at": _utc_now(), "records": records,
    })
    return records


def semantic_vector(profile: dict[str, Any]) -> dict[str, Any]:
    """Flatten a Pipeline 1 semantic profile without fabricating absent fields."""
    context = profile.get("dataset_context") or {}
    columns = profile.get("columns") or []
    count = max(1, len(columns))
    distributions: dict[str, dict[str, int]] = {}
    for field in ("semantic_type", "value_function", "numeric_meaning", "semantic_role"):
        values: dict[str, int] = {}
        for column in columns:
            value = column.get(field)
            values[str(value) if value is not None else "unknown"] = values.get(str(value) if value is not None else "unknown", 0) + 1
        distributions[field] = values
    return {
        "probable_domain": context.get("probable_domain"),
        "domain_confidence": context.get("domain_confidence"),
        "row_entity": context.get("row_entity"),
        "row_entity_confidence": context.get("row_entity_confidence"),
        "dataset_description_present": bool(context.get("dataset_description")),
        "column_count": len(columns),
        "mean_column_confidence": sum(float(c.get("confidence") or 0) for c in columns) / count,
        "distributions": distributions,
        "warning_count": len(profile.get("warnings") or []),
    }


class GroundworkOllamaProvider:
    """Ollama adapter matching production payloads with a runner-only timeout."""

    def __init__(self, model: str, timeout_seconds: float):
        self.model = model
        self.base_url = "http://localhost:11434"
        self.timeout_seconds = timeout_seconds
        self.last_response_metadata: dict[str, Any] | None = None

    def generate_structured(self, prompt: str, response_model: Any) -> dict[str, Any]:
        import requests

        payload = {
            "model": self.model, "prompt": prompt, "stream": False,
            "format": response_model.model_json_schema(),
            "options": {"temperature": 0},
        }
        self.last_response_metadata = None
        try:
            response = requests.post(
                f"{self.base_url}/api/generate", json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to communicate with Ollama: {exc}") from exc
        data = response.json()
        self.last_response_metadata = {
            key: data.get(key) for key in (
                "model", "created_at", "done", "done_reason", "total_duration",
                "load_duration", "prompt_eval_count", "prompt_eval_duration",
                "eval_count", "eval_duration",
            )
        }
        raw = data.get("response")
        if not raw:
            raise RuntimeError("Ollama returned an empty response.")
        return response_model.model_validate_json(raw).model_dump()


def run_semantic_profiles(
    manifest: list[dict[str, Any]], output_dir: Path, model: str,
    selected_task_ids: set[int] | None = None, llm_timeout: float = 240,
) -> list[dict[str, Any]]:
    """Run only the frozen Architecture-B profiler, resuming completed datasets."""
    from intelligence.rag.retriever import RAGRetriever
    from pipelines.pipeline1_cleaning.profiling.dataset_profiler import profile_dataset
    from pipelines.pipeline1_cleaning.semantics.batched_analyzer import analyze_batched_semantics
    from pipelines.pipeline1_cleaning.semantics.contradiction_guard import apply_semantic_contradiction_guard
    from pipelines.pipeline1_cleaning.semantics.evidence_builder import build_dataset_evidence

    profile_dir = output_dir / "semantic_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    retriever = RAGRetriever(knowledge_directory=str(ROOT / "knowledge_base"))
    retriever.index_knowledge()
    provider = GroundworkOllamaProvider(model=model, timeout_seconds=llm_timeout)
    summary = []
    for item in manifest:
        if selected_task_ids and int(item["task_id"]) not in selected_task_ids:
            continue
        destination = profile_dir / f"task_{item['task_id']}.json"
        if destination.is_file():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                summary.append({"dataset_id": str(item["task_id"]), "status": "already_complete"})
                continue
        source = DATASET_DIR / item["csv_file"]
        if not source.is_file():
            summary.append({"dataset_id": str(item["task_id"]), "status": "missing_local_csv"})
            continue
        try:
            frame = pd.read_csv(source, low_memory=False)
            deterministic = profile_dataset(frame)
            profile_model = analyze_batched_semantics(
                dataset_profile=deterministic,
                dataset_evidence=build_dataset_evidence(deterministic),
                retriever=retriever,
                llm_provider=provider,
            )
            profile = apply_semantic_contradiction_guard(
                profile_model, frame,
            ).model_dump(mode="json")
            vector = semantic_vector(profile)
            degraded = bool(profile.get("telemetry", {}).get("batch_failures")) or (
                vector["column_count"] > 0
                and vector["distributions"]["semantic_type"].get("unknown", 0)
                == vector["column_count"]
            )
            status = "degraded" if degraded else "complete"
            _atomic_json(destination, {
                "dataset_id": str(item["task_id"]), "dataset_name": item["dataset_name"],
                "model": model, "llm_timeout_seconds": llm_timeout,
                "generated_at": _utc_now(), "status": status,
                "semantic_profile": profile, "semantic_meta_features": vector,
            })
            summary.append({"dataset_id": str(item["task_id"]), "status": status})
        except Exception as exc:
            summary.append({
                "dataset_id": str(item["task_id"]), "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
        _atomic_json(output_dir / "semantic_run_status.json", {
            "updated_at": _utc_now(), "model": model, "records": summary,
        })
    return summary


def _normalized_task_type(task_type: str | None) -> str | None:
    if task_type is None:
        return None
    value = task_type.casefold()
    if "classification" in value:
        return "classification"
    if "regression" in value:
        return "regression"
    return None


def task_type_supported(model_id: str, task_type: str | None) -> bool | None:
    key = re.sub(r"[^a-z0-9]", "", model_id.casefold())
    capability = MODEL_CAPABILITIES.get(key)
    normalized = _normalized_task_type(task_type)
    if capability is None or normalized is None:
        return None
    return normalized in capability["task_type_supported"]


def feasibility(
    model_id: str,
    training_samples: int | None,
    feature_count: int | None,
    task_type: str | None = None,
    class_count: int | None = None,
) -> tuple[bool | None, str]:
    key = re.sub(r"[^a-z0-9]", "", model_id.casefold())
    supported = task_type_supported(model_id, task_type)
    if supported is False:
        return False, f"task_type={task_type!r} is not supported by {model_id}"
    if task_type is not None and supported is None:
        return None, f"Task type or model capability is not registered: {task_type!r}"

    capability = MODEL_CAPABILITIES.get(key)
    if capability is None:
        return None, "No deterministic feasibility rule is registered for this model."

    minimum_rows = capability["minimum_training_rows"]
    if minimum_rows is not None:
        if training_samples is None:
            return None, "Training-sample count is required for the registered minimum."
        if training_samples < minimum_rows:
            return False, f"training_samples={training_samples} is below {minimum_rows}"

    maximum_rows = capability["maximum_training_rows"]
    if maximum_rows is not None:
        if training_samples is None:
            return None, "Training-sample count is required for the registered maximum."
        if training_samples > maximum_rows:
            return False, f"training_samples={training_samples} exceeds {maximum_rows}"

    maximum_features = capability["maximum_features"]
    if maximum_features is not None:
        if feature_count is None:
            return None, "Feature count is required for the registered maximum."
        if feature_count > maximum_features:
            return False, f"feature_count={feature_count} exceeds {maximum_features}"

    maximum_classes = capability["maximum_classes"]
    if _normalized_task_type(task_type) == "classification" and maximum_classes is not None:
        if class_count is None:
            return None, "Class count is required for the registered maximum."
        if class_count > maximum_classes:
            return False, f"class_count={class_count} exceeds {maximum_classes}"

    return True, "Within all registered implementation-specific restrictions."


def compute_model_feasibility(
    manifest: list[dict[str, Any]],
    statistical_records: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Materialize deterministic model/dataset feasibility before evaluation."""

    statistics_by_id = {
        str(record["dataset_id"]): record.get("statistical_meta_features") or {}
        for record in statistical_records
    }
    records = []
    for dataset in manifest:
        dataset_id = str(dataset["task_id"])
        statistics = statistics_by_id.get(dataset_id, {})
        for model_id in EVALUATION_MODEL_IDS:
            supported = task_type_supported(model_id, dataset.get("task_type"))
            feasible, reason = feasibility(
                model_id,
                statistics.get("row_count"),
                statistics.get("feature_count"),
                task_type=dataset.get("task_type"),
                class_count=statistics.get("target_class_count"),
            )
            records.append({
                "dataset_id": dataset_id,
                "dataset_name": dataset["dataset_name"],
                "model_id": model_id,
                "task_type": dataset.get("task_type"),
                "task_type_supported": supported,
                "feasible": feasible,
                "feasibility_reason": reason,
            })
    _atomic_json(output_dir / "model_feasibility.json", {
        "generated_at": _utc_now(),
        "records": records,
    })
    return records


def validate_log_record(record: dict[str, Any]) -> None:
    missing = [field for field in LOG_COLUMNS if field not in record]
    if missing:
        raise ValueError("Missing model-run fields: " + ", ".join(missing))
    if record["source"] not in VALID_SOURCES:
        raise ValueError(f"Invalid source: {record['source']}")
    if not isinstance(record["feasible"], bool):
        raise ValueError("feasible must be a resolved boolean before a run is logged")
    if record["status"] not in {"success", "infeasible_skip", "failure"}:
        raise ValueError(f"Invalid model-run status: {record['status']}")
    if record["status"] == "infeasible_skip" and record["feasible"]:
        raise ValueError("infeasible_skip rows must have feasible=false")
    if record["status"] == "success" and not record["feasible"]:
        raise ValueError("success rows must have feasible=true")
    if not record["feasible"] and record["metric_value"] is not None:
        raise ValueError("Infeasible runs must not contain a performance score")
    if record["metric_value"] is not None and not record["metric_name"]:
        raise ValueError("A metric_value requires metric_name")


def initialize_log(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "model_run_log.csv"
    if log_path.exists():
        with log_path.open("r", encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.reader(handle))
        existing_header = existing_rows[0] if existing_rows else []
        if existing_header != LOG_COLUMNS:
            if len(existing_rows) <= 1:
                with log_path.open("w", encoding="utf-8", newline="") as handle:
                    csv.DictWriter(handle, fieldnames=LOG_COLUMNS).writeheader()
            else:
                raise ValueError(
                    "Existing model_run_log.csv uses an incompatible schema; "
                    "select a new output directory instead of mixing run formats"
                )
    else:
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=LOG_COLUMNS).writeheader()
    _atomic_json(output_dir / "model_run_log_schema.json", {
        "version": 2,
        "columns": LOG_COLUMNS,
        "valid_sources": sorted(VALID_SOURCES),
        "invariant": "Every dataset/model outcome is logged; infeasible rows have no metric",
        "foundation_model_limits": FOUNDATION_MODEL_LIMITS,
        "model_capabilities": MODEL_CAPABILITIES,
    })


def summarize_groundwork(output_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(output_dir)
    statistical = json.loads(
        (output_dir / "statistical_meta_features.json").read_text(encoding="utf-8")
    )["records"]
    profile_paths = sorted((output_dir / "semantic_profiles").glob("task_*.json")) \
        if (output_dir / "semantic_profiles").is_dir() else []
    profiles = [json.loads(path.read_text(encoding="utf-8")) for path in profile_paths]
    with (output_dir / "model_run_log.csv").open("r", encoding="utf-8", newline="") as handle:
        logged_runs = max(0, sum(1 for _ in csv.reader(handle)) - 1)
    inventory = json.loads(
        (ROOT / "benchmarks" / "tabarena" / "model_inventory.json").read_text(encoding="utf-8")
    )
    summary = {
        "generated_at": _utc_now(),
        "datasets": {
            "official_suite_tasks": len(manifest),
            "local_csvs": sum(bool(item["local_csv_present"]) for item in manifest),
            "statistical_meta_features_complete": sum(item["status"] == "complete" for item in statistical),
        },
        "semantic_meta_features": {
            "complete": sum(item.get("status") == "complete" for item in profiles),
            "degraded": sum(item.get("status") == "degraded" for item in profiles),
            "pending": len(manifest) - len(profiles),
            "profiles": [{
                "dataset_id": str(item["dataset_id"]), "dataset_name": item["dataset_name"],
                "status": item.get("status"),
                "probable_domain": item["semantic_meta_features"].get("probable_domain"),
                "dataset_description_present": item["semantic_meta_features"].get("dataset_description_present"),
            } for item in profiles],
        },
        "candidate_models": {
            "expected": inventory["candidate_count_expected"],
            "found": inventory["candidate_count_found"],
            "unresolved_slots": inventory["unresolved_candidate_slots"],
            "blocking_gaps": inventory["blocking_gaps"],
        },
        "model_run_log_rows": logged_runs,
        "router_training_started": False,
    }
    _atomic_json(output_dir / "task_b_summary.json", summary)
    return summary


def load_manifest(output_dir: Path) -> list[dict[str, Any]]:
    return json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))["datasets"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "refresh", "sync-missing", "semantic", "summarize"])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--llm-timeout", type=float, default=240)
    parser.add_argument("--task-id", type=int, action="append")
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = fetch_manifest(timeout=args.timeout)
        write_manifest(manifest, args.output_dir)
        statistics = compute_statistical_artifacts(manifest, args.output_dir)
        compute_model_feasibility(manifest, statistics, args.output_dir)
        initialize_log(args.output_dir)
        summary = {
            "generated_at": _utc_now(), "suite_tasks": len(manifest),
            "local_csvs_present": sum(item["local_csv_present"] for item in manifest),
            "statistical_complete": sum(item["status"] == "complete" for item in statistics),
            "statistical_gaps": [item for item in statistics if item["status"] != "complete"],
        }
        _atomic_json(args.output_dir / "groundwork_summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    manifest = load_manifest(args.output_dir)
    if args.command == "summarize":
        print(json.dumps(summarize_groundwork(args.output_dir), indent=2))
        return 0
    if args.command == "refresh":
        manifest = refresh_local_manifest_state(manifest)
        write_manifest(manifest, args.output_dir)
        statistics = compute_statistical_artifacts(manifest, args.output_dir)
        compute_model_feasibility(manifest, statistics, args.output_dir)
        initialize_log(args.output_dir)
        summary = {
            "generated_at": _utc_now(), "suite_tasks": len(manifest),
            "local_csvs_present": sum(item["local_csv_present"] for item in manifest),
            "statistical_complete": sum(item["status"] == "complete" for item in statistics),
            "statistical_gaps": [item for item in statistics if item["status"] != "complete"],
        }
        _atomic_json(args.output_dir / "groundwork_summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "sync-missing":
        results = sync_missing_datasets(manifest, timeout=args.timeout)
        _atomic_json(args.output_dir / "dataset_sync.json", {
            "generated_at": _utc_now(), "records": results,
        })
        print(json.dumps(results, indent=2))
        return 0 if not any(item["status"] == "error" for item in results) else 1
    summary = run_semantic_profiles(
        manifest, args.output_dir, args.model,
        selected_task_ids=set(args.task_id) if args.task_id else None,
        llm_timeout=args.llm_timeout,
    )
    print(json.dumps(summary, indent=2))
    return 0 if not any(item["status"] in {"error", "degraded"} for item in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
