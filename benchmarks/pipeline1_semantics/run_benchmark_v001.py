"""Run the frozen Pipeline 1 semantic architecture as Benchmark V001.

This is a diagnostic benchmark, not a ground-truth accuracy evaluation.

Run from the repository root:
    python benchmarks/pipeline1_semantics/run_benchmark_v001.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIRECTORY = PROJECT_ROOT / "tests" / "datasets" / "valid"
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "benchmarks"
    / "pipeline1_semantics"
    / "benchmark_v001_baseline"
)
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
BENCHMARK_VERSION = "benchmark_v001_baseline"
BENCHMARK_NAME = "LYNX Pipeline 1 semantic diagnostic baseline"
SEMANTIC_CONTRACT = {
    "column": "semantic_type + value_function + concept_description",
    "role": "referenced_entity + entity_alignment + semantic_role",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return make_json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return make_json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return make_json_safe(model.model_dump(mode="json"))
    if hasattr(model, "dict"):
        return make_json_safe(model.dict())
    if isinstance(model, dict):
        return make_json_safe(model)
    raise TypeError(f"Unsupported result type: {type(model).__name__}")


def atomic_write_json(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(make_json_safe(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(make_json_safe(value), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def timed_call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    try:
        return function(*args, **kwargs), time.perf_counter() - started
    except Exception as error:
        setattr(error, "benchmark_elapsed_seconds", time.perf_counter() - started)
        raise


class CallTimingRecorder:
    """Observe retriever/LLM calls without changing their inputs or outputs."""

    def __init__(self) -> None:
        self.retrieval_labels: list[str] = []
        self.retrieval_index = 0
        self.llm_label: str | None = None
        self.timings: Counter[str] = Counter()
        self.call_counts: Counter[str] = Counter()
        self.llm_call_metrics: dict[str, list[dict[str, Any]]] = {}

    def begin(self, retrieval_labels: list[str], llm_label: str) -> None:
        self.retrieval_labels = retrieval_labels
        self.retrieval_index = 0
        self.llm_label = llm_label
        self.timings.clear()
        self.call_counts.clear()
        self.llm_call_metrics.clear()

    def retrieval_label(self) -> str:
        if self.retrieval_index < len(self.retrieval_labels):
            label = self.retrieval_labels[self.retrieval_index]
        else:
            label = f"unexpected_rag_retrieval_{self.retrieval_index + 1}"
        self.retrieval_index += 1
        return label

    def record(self, label: str, elapsed: float) -> None:
        self.timings[label] += elapsed
        self.call_counts[label] += 1

    def finish(self) -> tuple[dict[str, float], dict[str, int]]:
        return (
            {key: round(value, 6) for key, value in self.timings.items()},
            dict(self.call_counts),
        )

    def record_llm_metrics(self, label: str, metrics: dict[str, Any]) -> None:
        self.llm_call_metrics.setdefault(label, []).append(metrics)

    def llm_metrics(self) -> dict[str, list[dict[str, Any]]]:
        return make_json_safe(self.llm_call_metrics)


class TimedRetriever:
    def __init__(self, wrapped: Any, recorder: CallTimingRecorder) -> None:
        self.wrapped = wrapped
        self.recorder = recorder

    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        label = self.recorder.retrieval_label()
        started = time.perf_counter()
        try:
            return self.wrapped.retrieve(*args, **kwargs)
        finally:
            self.recorder.record(label, time.perf_counter() - started)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)


class TimedLLMProvider:
    def __init__(self, wrapped: Any, recorder: CallTimingRecorder) -> None:
        self.wrapped = wrapped
        self.recorder = recorder

    def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
        label = self.recorder.llm_label or "unexpected_llm_inference"
        prompt = kwargs.get("prompt")
        if prompt is None and args:
            prompt = args[0]
        started = time.perf_counter()
        try:
            return self.wrapped.generate_structured(*args, **kwargs)
        finally:
            api_elapsed = time.perf_counter() - started
            self.recorder.record(label, api_elapsed)
            raw = getattr(self.wrapped, "last_response_metadata", None) or {}

            def seconds(nanoseconds: Any) -> float | None:
                return (
                    nanoseconds / 1_000_000_000
                    if isinstance(nanoseconds, (int, float))
                    else None
                )

            prompt_count = raw.get("prompt_eval_count")
            output_count = raw.get("eval_count")
            prompt_seconds = seconds(raw.get("prompt_eval_duration"))
            generation_seconds = seconds(raw.get("eval_duration"))
            total_seconds = seconds(raw.get("total_duration"))
            load_seconds = seconds(raw.get("load_duration"))
            self.recorder.record_llm_metrics(
                label,
                {
                    "prompt_character_length": len(prompt) if isinstance(prompt, str) else None,
                    "prompt_eval_count": prompt_count,
                    "prompt_eval_duration_seconds": prompt_seconds,
                    "prompt_eval_tokens_per_second": round(
                        prompt_count / prompt_seconds, 6
                    ) if prompt_count and prompt_seconds else None,
                    "output_token_count": output_count,
                    "eval_duration_seconds": generation_seconds,
                    "generation_tokens_per_second": round(
                        output_count / generation_seconds, 6
                    ) if output_count and generation_seconds else None,
                    "ollama_total_duration_seconds": total_seconds,
                    "ollama_load_duration_seconds": load_seconds,
                    "api_wall_duration_seconds": round(api_elapsed, 6),
                    "provider_api_overhead_seconds": round(
                        max(0.0, api_elapsed - total_seconds), 6
                    ) if total_seconds is not None else None,
                    "done_reason": raw.get("done_reason"),
                },
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)


def timed_semantic_operation(
    function: Callable[..., Any],
    recorder: CallTimingRecorder,
    retrieval_labels: list[str],
    llm_label: str,
    **kwargs: Any,
) -> tuple[
    Any,
    dict[str, float],
    dict[str, int],
    dict[str, list[dict[str, Any]]],
    float,
]:
    recorder.begin(retrieval_labels, llm_label)
    started = time.perf_counter()
    try:
        result = function(**kwargs)
    except Exception as error:
        setattr(error, "benchmark_elapsed_seconds", time.perf_counter() - started)
        timings, counts = recorder.finish()
        setattr(error, "benchmark_nested_timings", timings)
        setattr(error, "benchmark_nested_call_counts", counts)
        setattr(error, "benchmark_llm_call_metrics", recorder.llm_metrics())
        raise
    total = time.perf_counter() - started
    timings, counts = recorder.finish()
    return result, timings, counts, recorder.llm_metrics(), total


def discover_datasets(
    dataset_directory: Path,
    selected_names: set[str] | None = None,
) -> tuple[list[Path], list[dict[str, Any]], list[dict[str, Any]]]:
    if not dataset_directory.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_directory}"
        )

    files = sorted(
        (path for path in dataset_directory.iterdir() if path.is_file()),
        key=lambda path: path.name.casefold(),
    )
    discovered = [
        {
            "name": path.name,
            "path": str(path.resolve()),
            "extension": path.suffix.lower(),
            "supported": path.suffix.lower() in SUPPORTED_EXTENSIONS,
        }
        for path in files
    ]

    supported = [
        path for path in files if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    skipped = [
        {
            "name": path.name,
            "path": str(path.resolve()),
            "reason": f"unsupported extension: {path.suffix.lower() or '(none)'}",
        }
        for path in files
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS
    ]

    if selected_names:
        available_names = {path.name for path in supported}
        missing = sorted(selected_names - available_names)
        if missing:
            raise ValueError(
                "Selected dataset files were not found or supported: "
                + ", ".join(missing)
            )
        supported = [path for path in supported if path.name in selected_names]

    return supported, discovered, skipped


def create_pipeline_runtime(model_name: str) -> dict[str, Any]:
    """Import and initialize the current real Pipeline 1 components."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from intelligence.llm.ollama_provider import OllamaProvider
    from intelligence.rag.retriever import RAGRetriever
    from pipelines.pipeline1_cleaning.ingestion.dataset_loader import load_dataset
    from pipelines.pipeline1_cleaning.profiling.dataset_profiler import profile_dataset
    from pipelines.pipeline1_cleaning.semantics.column_understanding import (
        understand_column,
    )
    from pipelines.pipeline1_cleaning.semantics.dataset_understanding import (
        understand_dataset,
    )
    from pipelines.pipeline1_cleaning.semantics.evidence_builder import (
        build_column_evidence,
        build_dataset_evidence,
    )
    from pipelines.pipeline1_cleaning.semantics.role_resolver import (
        resolve_column_role,
    )

    retriever = RAGRetriever(
        knowledge_directory=str(PROJECT_ROOT / "knowledge_base")
    )
    indexing_started = time.perf_counter()
    retriever.index_knowledge()
    indexing_seconds = time.perf_counter() - indexing_started

    return {
        "load_dataset": load_dataset,
        "profile_dataset": profile_dataset,
        "build_dataset_evidence": build_dataset_evidence,
        "build_column_evidence": build_column_evidence,
        "understand_dataset": understand_dataset,
        "understand_column": understand_column,
        "resolve_column_role": resolve_column_role,
        "retriever": retriever,
        "llm_provider": OllamaProvider(model=model_name),
        "indexed_chunks": retriever.vector_store.count(),
        "embedding_model": getattr(retriever.embedding_model, "model_name", None),
        "knowledge_indexing_seconds": round(indexing_seconds, 6),
    }


def is_unknown(value: Any) -> bool:
    return value is None or str(value).strip().casefold() in {"", "unknown", "none"}


def run_consistency_checks(
    column_semantics: dict[str, Any],
    role_resolution: dict[str, Any],
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    def add(severity: str, rule: str, message: str) -> None:
        checks.append({"severity": severity, "rule": rule, "message": message})

    alignment = role_resolution.get("entity_alignment")
    role = role_resolution.get("semantic_role")
    referenced_entity = role_resolution.get("referenced_entity")
    value_function = column_semantics.get("value_function")
    semantic_type = column_semantics.get("semantic_type")
    semantic_confidence = column_semantics.get("confidence")
    role_confidence = role_resolution.get("confidence")

    if alignment == "same_as_row_entity" and role == "foreign_identifier":
        add("hard_contradiction", "same_entity_foreign_role",
            "Same-row-entity alignment conflicts with foreign_identifier.")
    if alignment == "different_from_row_entity" and role == "primary_identifier":
        add("hard_contradiction", "different_entity_primary_role",
            "Different-entity alignment conflicts with primary_identifier.")
    if alignment == "not_applicable" and referenced_entity is not None:
        add("hard_contradiction", "not_applicable_with_entity",
            "Not-applicable alignment requires a null referenced entity.")
    if alignment in {"same_as_row_entity", "different_from_row_entity"} and not referenced_entity:
        add("hard_contradiction", "resolved_alignment_without_entity",
            "Resolved entity alignment requires a referenced entity.")

    identifier_roles = {
        "primary_identifier", "foreign_identifier", "transaction_identifier"
    }
    if value_function == "entity_reference" and semantic_type == "unknown":
        add("soft_warning", "entity_reference_unknown_type",
            "Entity reference has unknown semantic type.")
    if value_function == "entity_reference" and role not in identifier_roles:
        add("soft_warning", "entity_reference_non_identifier_role",
            "Entity reference resolved to a non-identifier role.")
    if value_function == "classification" and role in {
        "primary_identifier", "foreign_identifier"
    }:
        add("soft_warning", "classification_identifier_role",
            "Classification resolved to a primary or foreign identifier role.")
    if (
        is_unknown(semantic_type) or is_unknown(value_function)
    ) and isinstance(semantic_confidence, (int, float)) and semantic_confidence >= 0.9:
        add("soft_warning", "high_confidence_unknown_semantics",
            "Column semantics contain an unknown value at high confidence.")
    if is_unknown(role) and isinstance(role_confidence, (int, float)) and role_confidence >= 0.9:
        add("soft_warning", "high_confidence_unknown_role",
            "Semantic role is unknown at high confidence.")

    return checks


class BenchmarkStatistics:
    def __init__(self, datasets_discovered: int) -> None:
        self.started = time.monotonic()
        self.counts = Counter({"datasets_discovered": datasets_discovered})
        self.semantic_types: Counter[str] = Counter()
        self.value_functions: Counter[str] = Counter()
        self.semantic_roles: Counter[str] = Counter()
        self.entity_alignments: Counter[str] = Counter()
        self.warning_rules: Counter[str] = Counter()
        self.semantic_confidences: list[float] = []
        self.role_confidences: list[float] = []
        self.dataset_durations: list[float] = []
        self.column_durations: list[float] = []
        self.stage_timings: Counter[str] = Counter()
        self.stage_counts: Counter[str] = Counter()

    def record_timings(self, timings: dict[str, float]) -> None:
        for stage, elapsed in timings.items():
            self.stage_timings[stage] += elapsed
            self.stage_counts[stage] += 1

    @staticmethod
    def confidence_summary(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"average": None, "minimum": None, "maximum": None}
        return {
            "average": round(sum(values) / len(values), 6),
            "minimum": min(values),
            "maximum": max(values),
        }

    def summary(self) -> dict[str, Any]:
        columns_successful = self.counts["columns_successful"]
        type_unknown = self.semantic_types["unknown"]
        function_unknown = self.value_functions["unknown"]
        role_unknown = self.semantic_roles["unknown"]

        def rate(count: int) -> float:
            return round(count / columns_successful, 6) if columns_successful else 0.0

        return {
            "benchmark_version": BENCHMARK_VERSION,
            "generated_at": utc_now(),
            "run": {
                key: self.counts[key]
                for key in (
                    "datasets_discovered", "datasets_processed",
                    "datasets_successful", "datasets_partial", "datasets_failed",
                    "columns_attempted", "columns_successful", "columns_failed",
                )
            },
            "dataset_semantics": {
                "dataset_domain_unknown_count": self.counts["dataset_domain_unknown"],
                "row_entity_unknown_count": self.counts["row_entity_unknown"],
            },
            "column_semantics": {
                "semantic_type_distribution": dict(sorted(self.semantic_types.items())),
                "semantic_type_unknown_count": type_unknown,
                "semantic_type_unknown_rate": rate(type_unknown),
                "value_function_distribution": dict(sorted(self.value_functions.items())),
                "value_function_unknown_count": function_unknown,
                "value_function_unknown_rate": rate(function_unknown),
                "semantic_role_distribution": dict(sorted(self.semantic_roles.items())),
                "semantic_role_unknown_count": role_unknown,
                "semantic_role_unknown_rate": rate(role_unknown),
                "entity_alignment_distribution": dict(sorted(self.entity_alignments.items())),
            },
            "confidence": {
                "semantic": self.confidence_summary(self.semantic_confidences),
                "role": self.confidence_summary(self.role_confidences),
            },
            "consistency": {
                "hard_contradiction_count": self.counts["hard_contradictions"],
                "soft_warning_count": self.counts["soft_warnings"],
                "warning_distribution_by_rule": dict(sorted(self.warning_rules.items())),
            },
            "runtime": {
                "total_elapsed_seconds": round(time.monotonic() - self.started, 3),
                "average_seconds_per_dataset": round(
                    sum(self.dataset_durations) / len(self.dataset_durations), 3
                ) if self.dataset_durations else None,
                "average_seconds_per_column": round(
                    sum(self.column_durations) / len(self.column_durations), 3
                ) if self.column_durations else None,
                "stage_timings": {
                    stage: {
                        "total_seconds": round(total, 6),
                        "observations": self.stage_counts[stage],
                        "average_seconds": round(
                            total / self.stage_counts[stage], 6
                        ),
                    }
                    for stage, total in sorted(self.stage_timings.items())
                },
            },
            "methodology_note": (
                "Diagnostic coverage and consistency statistics only; no human "
                "ground-truth accuracy metrics are reported."
            ),
        }


def error_record(
    scope: str,
    dataset_name: str | None,
    column_name: str | None,
    stage: str,
    error: Exception,
    partial_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "dataset_name": dataset_name,
        "column_name": column_name,
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
        "timestamp": utc_now(),
        "partial_result": make_json_safe(partial_result),
    }


def physical_evidence(column: dict[str, Any]) -> dict[str, Any]:
    return {
        "physical_dtype": column.get("physical_dtype"),
        "missing_percentage": column.get("missing", {}).get("percentage"),
        "unique_count": column.get("uniqueness", {}).get("unique_count"),
        "unique_ratio": column.get("uniqueness", {}).get("unique_ratio"),
        "repeated_occurrences": column.get("repetition", {}).get(
            "repeated_occurrences"
        ),
        "repetition_rate": column.get("repetition", {}).get("repetition_rate"),
        "sample_values": make_json_safe(column.get("sample_values", [])),
        "numeric_statistics": make_json_safe(column.get("numeric_statistics")),
    }


def prepare_output_directory(output_directory: Path) -> dict[str, Path]:
    if output_directory.exists():
        raise FileExistsError(
            f"Benchmark destination already exists and will not be overwritten: "
            f"{output_directory}"
        )
    output_directory.mkdir(parents=True)
    paths = {
        name: output_directory / name
        for name in (
            "manifest.json", "datasets.jsonl", "columns.jsonl",
            "errors.jsonl", "summary.json",
        )
    }
    for name in ("datasets.jsonl", "columns.jsonl", "errors.jsonl"):
        paths[name].touch()
    return paths


def run_benchmark(
    dataset_directory: Path,
    output_directory: Path,
    selected_names: set[str] | None = None,
    selected_columns: set[str] | None = None,
    profile_timing: bool = False,
    model_name: str = "qwen2.5:3b",
    dataset_rag_top_k: int = 8,
    column_rag_top_k: int = 6,
    role_rag_top_k: int = 5,
    runtime_factory: Callable[[str], dict[str, Any]] = create_pipeline_runtime,
) -> int:
    benchmark_started = time.monotonic()
    datasets, discovered, skipped = discover_datasets(
        dataset_directory, selected_names
    )
    paths = prepare_output_directory(output_directory)
    stats = BenchmarkStatistics(len(datasets))

    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_name": BENCHMARK_NAME,
        "started_at": utc_now(),
        "completed_at": None,
        "status": "initializing",
        "dataset_directory": str(dataset_directory.resolve()),
        "discovered_files": discovered,
        "processed_files": [],
        "skipped_files": skipped,
        "configuration": {
            "llm_provider": "OllamaProvider",
            "llm_model": model_name,
            "embedding_model": None,
            "dataset_rag_top_k": dataset_rag_top_k,
            "column_rag_top_k": column_rag_top_k,
            "role_rag_top_k": role_rag_top_k,
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
            "selected_columns": sorted(selected_columns) if selected_columns else None,
            "profile_timing": profile_timing,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "semantic_contract": SEMANTIC_CONTRACT,
        "methodology": "diagnostic benchmark without human ground truth",
    }
    atomic_write_json(paths["manifest.json"], manifest)
    atomic_write_json(paths["summary.json"], stats.summary())

    try:
        runtime = runtime_factory(model_name)
    except Exception as error:
        append_jsonl(
            paths["errors.jsonl"],
            error_record("run", None, None, "runtime_initialization", error),
        )
        manifest["status"] = "failed_initialization"
        manifest["completed_at"] = utc_now()
        atomic_write_json(paths["manifest.json"], manifest)
        atomic_write_json(paths["summary.json"], stats.summary())
        return 1

    manifest["configuration"]["embedding_model"] = runtime.get("embedding_model")
    manifest["configuration"]["indexed_chunks"] = runtime.get("indexed_chunks")
    manifest["configuration"]["knowledge_indexing_seconds"] = runtime.get(
        "knowledge_indexing_seconds"
    )
    timing_recorder = CallTimingRecorder()
    timed_retriever = TimedRetriever(runtime["retriever"], timing_recorder)
    timed_llm_provider = TimedLLMProvider(
        runtime["llm_provider"], timing_recorder
    )
    manifest["status"] = "running"
    atomic_write_json(paths["manifest.json"], manifest)

    total_datasets = len(datasets)
    for dataset_index, dataset_path in enumerate(datasets, start=1):
        dataset_started = time.monotonic()
        dataset_name = dataset_path.name
        manifest["processed_files"].append(dataset_name)
        stats.counts["datasets_processed"] += 1
        print(f"[{dataset_index}/{total_datasets}] {dataset_name}", flush=True)

        dataset_record: dict[str, Any] = {
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path.resolve()),
            "status": "failed",
            "row_count": None,
            "column_count": None,
            "dataset_semantics": None,
            "columns_attempted": 0,
            "columns_succeeded": 0,
            "columns_failed": 0,
            "elapsed_seconds": None,
            "stage_timings_seconds": {},
            "stage_call_counts": {},
            "llm_call_metrics": {},
        }
        dataset_timings: dict[str, float] = {}
        dataset_call_counts: dict[str, int] = {}
        dataset_llm_metrics: dict[str, list[dict[str, Any]]] = {}

        dataset_stage = "dataset_loading"
        try:
            loaded, elapsed = timed_call(
                runtime["load_dataset"], str(dataset_path)
            )
            dataset_timings["dataset_loading"] = round(elapsed, 6)
            dataframe = loaded["dataframe"]
            dataset_record["row_count"] = int(dataframe.shape[0])
            dataset_record["column_count"] = int(dataframe.shape[1])
            dataset_stage = "profiling"
            dataset_profile, elapsed = timed_call(
                runtime["profile_dataset"], dataframe
            )
            dataset_timings["profiling"] = round(elapsed, 6)
            dataset_stage = "dataset_evidence"
            dataset_evidence, elapsed = timed_call(
                runtime["build_dataset_evidence"], dataset_profile
            )
            dataset_timings["dataset_evidence_building"] = round(elapsed, 6)
            dataset_stage = "dataset_understanding"
            (
                dataset_semantics_model,
                nested_timings,
                nested_counts,
                nested_llm_metrics,
                understanding_total,
            ) = timed_semantic_operation(
                runtime["understand_dataset"],
                timing_recorder,
                ["dataset_rag_retrieval"],
                "dataset_understanding_llm_inference",
                evidence=dataset_evidence,
                retriever=timed_retriever,
                llm_provider=timed_llm_provider,
                top_k=dataset_rag_top_k,
            )
            dataset_timings.update(nested_timings)
            dataset_call_counts.update(nested_counts)
            dataset_llm_metrics.update(nested_llm_metrics)
            dataset_timings["dataset_understanding_total"] = round(
                understanding_total, 6
            )
            dataset_timings["dataset_understanding_overhead"] = round(
                max(0.0, understanding_total - sum(nested_timings.values())), 6
            )
            dataset_semantics = model_to_dict(dataset_semantics_model)
            dataset_record["dataset_semantics"] = dataset_semantics
            if is_unknown(dataset_semantics.get("dataset_domain")):
                stats.counts["dataset_domain_unknown"] += 1
            if is_unknown(dataset_semantics.get("row_entity")):
                stats.counts["row_entity_unknown"] += 1
        except Exception as error:
            elapsed = getattr(error, "benchmark_elapsed_seconds", None)
            if elapsed is not None:
                dataset_timings[dataset_stage] = round(elapsed, 6)
            dataset_timings.update(
                getattr(error, "benchmark_nested_timings", {})
            )
            dataset_call_counts.update(
                getattr(error, "benchmark_nested_call_counts", {})
            )
            dataset_llm_metrics.update(
                getattr(error, "benchmark_llm_call_metrics", {})
            )
            dataset_record["stage_timings_seconds"] = dataset_timings
            dataset_record["stage_call_counts"] = dataset_call_counts
            dataset_record["llm_call_metrics"] = dataset_llm_metrics
            stats.record_timings(dataset_timings)
            append_jsonl(
                paths["errors.jsonl"],
                error_record("dataset", dataset_name, None,
                             dataset_stage, error),
            )
            stats.counts["datasets_failed"] += 1
            elapsed = round(time.monotonic() - dataset_started, 3)
            dataset_record["elapsed_seconds"] = elapsed
            stats.dataset_durations.append(elapsed)
            append_jsonl(paths["datasets.jsonl"], dataset_record)
            atomic_write_json(paths["summary.json"], stats.summary())
            atomic_write_json(paths["manifest.json"], manifest)
            print("  DATASET FAILED", flush=True)
            continue

        dataset_record["stage_timings_seconds"] = dataset_timings
        dataset_record["stage_call_counts"] = dataset_call_counts
        dataset_record["llm_call_metrics"] = dataset_llm_metrics
        stats.record_timings(dataset_timings)
        if profile_timing:
            dataset_total = time.monotonic() - dataset_started
            print("  Dataset-stage timing:", flush=True)
            for stage, stage_elapsed in dataset_timings.items():
                if stage.endswith("_total"):
                    continue
                percentage = (
                    stage_elapsed / dataset_total * 100 if dataset_total else 0.0
                )
                print(
                    f"      {stage}: {stage_elapsed:.6f}s ({percentage:.1f}%)",
                    flush=True,
                )
            print(
                "      observed calls: "
                + json.dumps(dataset_call_counts, sort_keys=True),
                flush=True,
            )
            for label, calls in dataset_llm_metrics.items():
                for call in calls:
                    print(
                        f"      {label} metrics: "
                        + json.dumps(call, sort_keys=True),
                        flush=True,
                    )
        column_profiles = dataset_profile["column_profiles"]
        if selected_columns:
            available_columns = {
                profile["column_name"] for profile in column_profiles
            }
            missing_columns = selected_columns - available_columns
            if missing_columns:
                append_jsonl(
                    paths["errors.jsonl"],
                    {
                        "scope": "dataset",
                        "dataset_name": dataset_name,
                        "column_name": None,
                        "stage": "column_selection",
                        "error_type": "MissingSelectedColumns",
                        "message": "Selected columns not present: "
                        + ", ".join(sorted(missing_columns)),
                        "traceback": None,
                        "timestamp": utc_now(),
                        "partial_result": None,
                    },
                )
            column_profiles = [
                profile for profile in column_profiles
                if profile["column_name"] in selected_columns
            ]
        total_columns = len(column_profiles)
        for column_index, column_profile in enumerate(column_profiles, start=1):
            column_started = time.monotonic()
            column_name = column_profile["column_name"]
            dataset_record["columns_attempted"] += 1
            stats.counts["columns_attempted"] += 1
            partial: dict[str, Any] = {}
            column_timings: dict[str, float] = {}
            column_call_counts: dict[str, int] = {}
            column_llm_metrics: dict[str, list[dict[str, Any]]] = {}

            column_stage = "column_evidence"
            try:
                evidence_packet, elapsed = timed_call(
                    runtime["build_column_evidence"],
                    dataset_profile=dataset_profile,
                    column_profile=column_profile,
                )
                column_timings["column_evidence_building"] = round(elapsed, 6)
                column_evidence = evidence_packet["column"]
                column_stage = "column_understanding"
                (
                    column_semantics_model,
                    nested_timings,
                    nested_counts,
                    nested_llm_metrics,
                    semantic_total,
                ) = timed_semantic_operation(
                    runtime["understand_column"],
                    timing_recorder,
                    [
                        "semantic_context_rag_retrieval",
                        "value_pattern_rag_retrieval",
                    ],
                    "column_understanding_llm_inference",
                    column_evidence=column_evidence,
                    dataset_semantics=dataset_semantics_model,
                    retriever=timed_retriever,
                    llm_provider=timed_llm_provider,
                    top_k=column_rag_top_k,
                )
                column_timings.update(nested_timings)
                column_call_counts.update(nested_counts)
                column_llm_metrics.update(nested_llm_metrics)
                column_timings["column_understanding_total"] = round(
                    semantic_total, 6
                )
                column_timings["column_understanding_overhead"] = round(
                    max(0.0, semantic_total - sum(nested_timings.values())), 6
                )
                column_semantics = model_to_dict(column_semantics_model)
                partial["column_semantics"] = column_semantics
                column_stage = "role_resolution"
                (
                    role_model,
                    nested_timings,
                    nested_counts,
                    nested_llm_metrics,
                    role_total,
                ) = timed_semantic_operation(
                    runtime["resolve_column_role"],
                    timing_recorder,
                    ["role_resolver_rag_retrieval"],
                    "role_resolver_llm_inference",
                    column_evidence=column_evidence,
                    column_semantics=column_semantics_model,
                    dataset_semantics=dataset_semantics_model,
                    retriever=timed_retriever,
                    llm=timed_llm_provider,
                    top_k=role_rag_top_k,
                )
                column_timings.update(nested_timings)
                column_call_counts.update(nested_counts)
                column_llm_metrics.update(nested_llm_metrics)
                column_timings["role_resolution_total"] = round(role_total, 6)
                column_timings["role_resolution_overhead"] = round(
                    max(0.0, role_total - sum(nested_timings.values())), 6
                )
                role_resolution = model_to_dict(role_model)
                checks_started = time.perf_counter()
                checks = run_consistency_checks(column_semantics, role_resolution)
                column_timings["consistency_checks"] = round(
                    time.perf_counter() - checks_started, 6
                )
                elapsed_before_output = time.monotonic() - column_started
                column_timings["total_before_output"] = round(
                    elapsed_before_output, 6
                )

                output_started = time.perf_counter()
                append_jsonl(
                    paths["columns.jsonl"],
                    {
                        "dataset_name": dataset_name,
                        "column_name": column_name,
                        "physical_evidence": physical_evidence(column_evidence),
                        "dataset_context": {
                            "dataset_domain": dataset_semantics.get("dataset_domain"),
                            "row_entity": dataset_semantics.get("row_entity"),
                        },
                        "column_semantics": column_semantics,
                        "role_resolution": role_resolution,
                        "consistency_checks": checks,
                        "stage_timings_seconds": column_timings,
                        "stage_call_counts": column_call_counts,
                        "llm_call_metrics": column_llm_metrics,
                        "elapsed_seconds_before_output": round(
                            elapsed_before_output, 6
                        ),
                    },
                )
                serialization_write_seconds = time.perf_counter() - output_started
                column_timings["serialization_write_overhead"] = round(
                    serialization_write_seconds, 6
                )
                dataset_record["columns_succeeded"] += 1
                stats.counts["columns_successful"] += 1
                total_elapsed = time.monotonic() - column_started
                column_timings["total_column_elapsed"] = round(total_elapsed, 6)
                stats.column_durations.append(total_elapsed)
                stats.record_timings(column_timings)

                semantic_type = column_semantics.get("semantic_type") or "unknown"
                value_function = column_semantics.get("value_function") or "unknown"
                semantic_role = role_resolution.get("semantic_role") or "unknown"
                alignment = role_resolution.get("entity_alignment") or "unknown"
                stats.semantic_types[str(semantic_type)] += 1
                stats.value_functions[str(value_function)] += 1
                stats.semantic_roles[str(semantic_role)] += 1
                stats.entity_alignments[str(alignment)] += 1
                if isinstance(column_semantics.get("confidence"), (int, float)):
                    stats.semantic_confidences.append(column_semantics["confidence"])
                if isinstance(role_resolution.get("confidence"), (int, float)):
                    stats.role_confidences.append(role_resolution["confidence"])
                for check in checks:
                    stats.warning_rules[check["rule"]] += 1
                    if check["severity"] == "hard_contradiction":
                        stats.counts["hard_contradictions"] += 1
                    else:
                        stats.counts["soft_warnings"] += 1
                status = "OK"
            except Exception as error:
                elapsed = getattr(error, "benchmark_elapsed_seconds", None)
                if elapsed is not None:
                    column_timings[column_stage] = round(elapsed, 6)
                column_timings.update(
                    getattr(error, "benchmark_nested_timings", {})
                )
                column_call_counts.update(
                    getattr(error, "benchmark_nested_call_counts", {})
                )
                column_llm_metrics.update(
                    getattr(error, "benchmark_llm_call_metrics", {})
                )
                column_timings["total_column_elapsed"] = round(
                    time.monotonic() - column_started, 6
                )
                partial["stage_timings_seconds"] = column_timings
                partial["stage_call_counts"] = column_call_counts
                partial["llm_call_metrics"] = column_llm_metrics
                stats.record_timings(column_timings)
                dataset_record["columns_failed"] += 1
                stats.counts["columns_failed"] += 1
                append_jsonl(
                    paths["errors.jsonl"],
                    error_record(
                        "column", dataset_name, column_name,
                        column_stage, error, partial,
                    ),
                )
                status = "FAILED"

            progress = f"  [{column_index}/{total_columns}] {column_name} ... {status}"
            print(progress, flush=True)
            if profile_timing:
                total = column_timings.get("total_column_elapsed", 0.0)
                for stage, stage_elapsed in column_timings.items():
                    if stage.endswith("_total") or stage in {
                        "total_before_output", "total_column_elapsed"
                    }:
                        continue
                    percentage = (stage_elapsed / total * 100) if total else 0.0
                    print(
                        f"      {stage}: {stage_elapsed:.6f}s ({percentage:.1f}%)",
                        flush=True,
                    )
                print(
                    f"      total_column_elapsed: {total:.6f}s (100.0%)",
                    flush=True,
                )
                print(
                    "      observed calls: "
                    + json.dumps(column_call_counts, sort_keys=True),
                    flush=True,
                )
                for label, calls in column_llm_metrics.items():
                    for call in calls:
                        print(
                            f"      {label} metrics: "
                            + json.dumps(call, sort_keys=True),
                            flush=True,
                        )
            atomic_write_json(paths["summary.json"], stats.summary())

        dataset_record["status"] = (
            "success" if dataset_record["columns_failed"] == 0 else "partial"
        )
        if dataset_record["status"] == "success":
            stats.counts["datasets_successful"] += 1
        else:
            stats.counts["datasets_partial"] += 1
        elapsed = round(time.monotonic() - dataset_started, 3)
        dataset_record["elapsed_seconds"] = elapsed
        stats.dataset_durations.append(elapsed)
        append_jsonl(paths["datasets.jsonl"], dataset_record)
        atomic_write_json(paths["summary.json"], stats.summary())
        atomic_write_json(paths["manifest.json"], manifest)

    manifest["completed_at"] = utc_now()
    manifest["status"] = (
        "completed" if not stats.counts["datasets_failed"]
        and not stats.counts["columns_failed"] else "completed_with_failures"
    )
    manifest["elapsed_seconds"] = round(time.monotonic() - benchmark_started, 3)
    atomic_write_json(paths["manifest.json"], manifest)
    atomic_write_json(paths["summary.json"], stats.summary())
    return 0 if manifest["status"] == "completed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=DEFAULT_DATASET_DIRECTORY
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument(
        "--dataset", action="append", dest="datasets",
        help="Process only this exact filename; repeat for multiple files.",
    )
    parser.add_argument(
        "--column", action="append", dest="columns",
        help="Process only this exact column name; repeat for multiple columns.",
    )
    parser.add_argument(
        "--profile-timing", action="store_true",
        help="Print detailed per-stage timing and percentages.",
    )
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--dataset-rag-top-k", type=int, default=8)
    parser.add_argument("--column-rag-top-k", type=int, default=6)
    parser.add_argument("--role-rag-top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_benchmark(
            dataset_directory=args.dataset_dir,
            output_directory=args.output_dir,
            selected_names=set(args.datasets) if args.datasets else None,
            selected_columns=set(args.columns) if args.columns else None,
            profile_timing=args.profile_timing,
            model_name=args.model,
            dataset_rag_top_k=args.dataset_rag_top_k,
            column_rag_top_k=args.column_rag_top_k,
            role_rag_top_k=args.role_rag_top_k,
        )
    except Exception as error:
        print(f"Benchmark could not start: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
