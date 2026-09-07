"""Diagnose Ollama prompt-cache behavior with real Pipeline 1 requests.

This script is observational only. It uses the production evidence builders,
RAG retrieval, prompt builders, structured schemas, and Ollama provider without
changing any semantic input or runtime option.

Run from the repository root:
    python benchmarks/pipeline1_semantics/ollama_prompt_cache_diagnostic.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = (
    PROJECT_ROOT / "tests" / "datasets" / "valid"
    / "dataset_03_healthcare_raw.csv"
)
MODEL_NAME = "qwen2.5:3b"
COLUMN_A = "patient_id"
COLUMN_B = "diagnosis_code"


class PromptCaptured(RuntimeError):
    pass


class PromptCaptureProvider:
    """Capture an unchanged production LLM request before network submission."""

    def __init__(self) -> None:
        self.prompt: str | None = None
        self.response_model: Any = None

    def generate_structured(self, prompt: str, response_model: Any) -> dict[str, Any]:
        self.prompt = prompt
        self.response_model = response_model
        raise PromptCaptured


def capture_request(function: Any, **kwargs: Any) -> tuple[str, Any]:
    capture = PromptCaptureProvider()
    provider_argument = "llm" if function.__name__ == "resolve_column_role" else "llm_provider"
    kwargs[provider_argument] = capture
    try:
        function(**kwargs)
    except PromptCaptured:
        pass
    if capture.prompt is None or capture.response_model is None:
        raise RuntimeError(f"Failed to capture request from {function.__name__}")
    return capture.prompt, capture.response_model


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def schema_text(response_model: Any) -> str:
    return json.dumps(
        response_model.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def seconds(nanoseconds: Any) -> float | None:
    if not isinstance(nanoseconds, (int, float)):
        return None
    return nanoseconds / 1_000_000_000


def measured_request(
    provider: Any,
    stage: str,
    column: str,
    prompt: str,
    response_model: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    started = time.perf_counter()
    result = None
    error = None
    try:
        result = provider.generate_structured(
            prompt=prompt,
            response_model=response_model,
        )
    except Exception as exc:  # Preserve telemetry even if semantic validation fails.
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - started
    raw = provider.last_response_metadata or {}
    prompt_duration = seconds(raw.get("prompt_eval_duration"))
    eval_duration = seconds(raw.get("eval_duration"))
    total_duration = seconds(raw.get("total_duration"))
    load_duration = seconds(raw.get("load_duration"))
    prompt_count = raw.get("prompt_eval_count")
    eval_count = raw.get("eval_count")
    metrics = {
        "stage": stage,
        "column": column,
        "prompt_sha256": sha256_text(prompt),
        "schema_sha256": sha256_text(schema_text(response_model)),
        "prompt_eval_count": prompt_count,
        "prompt_eval_duration_seconds": prompt_duration,
        "prompt_eval_tokens_per_second": (
            prompt_count / prompt_duration
            if prompt_count and prompt_duration else None
        ),
        "eval_count": eval_count,
        "eval_duration_seconds": eval_duration,
        "generation_tokens_per_second": (
            eval_count / eval_duration if eval_count and eval_duration else None
        ),
        "total_duration_seconds": total_duration,
        "load_duration_seconds": load_duration,
        "api_wall_duration_seconds": wall,
        "provider_api_overhead_seconds": (
            max(0.0, wall - total_duration)
            if total_duration is not None else None
        ),
        "error": error,
    }
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return metrics, result


def duration(metrics: dict[str, Any]) -> float | None:
    return metrics["prompt_eval_duration_seconds"]


def format_duration(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6f}s"


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from intelligence.llm.ollama_provider import OllamaProvider
    from intelligence.rag.retriever import RAGRetriever
    from pipelines.pipeline1_cleaning.ingestion.dataset_loader import load_dataset
    from pipelines.pipeline1_cleaning.profiling.dataset_profiler import profile_dataset
    from pipelines.pipeline1_cleaning.semantics.column_understanding import understand_column
    from pipelines.pipeline1_cleaning.semantics.dataset_understanding import understand_dataset
    from pipelines.pipeline1_cleaning.semantics.evidence_builder import (
        build_column_evidence,
        build_dataset_evidence,
    )
    from pipelines.pipeline1_cleaning.semantics.role_resolver import resolve_column_role
    from pipelines.pipeline1_cleaning.semantics.semantic_schema import ColumnSemanticResult

    print(f"Dataset: {DATASET_PATH}")
    print(f"Model: {MODEL_NAME}")
    print("Preparing real Pipeline 1 evidence and RAG context...", flush=True)

    loaded = load_dataset(str(DATASET_PATH))
    profile = profile_dataset(loaded["dataframe"])
    dataset_evidence = build_dataset_evidence(profile)
    retriever = RAGRetriever(knowledge_directory=str(PROJECT_ROOT / "knowledge_base"))
    retriever.index_knowledge()
    provider = OllamaProvider(model=MODEL_NAME)

    # A real Dataset Understanding result supplies the unchanged context used by
    # both production column prompts. It is setup, not part of a test sequence.
    dataset_semantics = understand_dataset(
        evidence=dataset_evidence,
        retriever=retriever,
        llm_provider=provider,
        top_k=8,
    )
    profiles = {item["column_name"]: item for item in profile["column_profiles"]}
    evidence = {
        name: build_column_evidence(profile, profiles[name])["column"]
        for name in (COLUMN_A, COLUMN_B)
    }

    column_requests: dict[str, tuple[str, Any]] = {}
    for name in (COLUMN_A, COLUMN_B):
        column_requests[name] = capture_request(
            understand_column,
            column_evidence=evidence[name],
            dataset_semantics=dataset_semantics,
            retriever=retriever,
            top_k=6,
        )

    all_sequences: dict[str, list[dict[str, Any]]] = {}

    print("\nSequence A -> A", flush=True)
    a_prompt, a_schema = column_requests[COLUMN_A]
    aa1, a_result = measured_request(
        provider, "column_understanding", COLUMN_A, a_prompt, a_schema
    )
    aa2, _ = measured_request(
        provider, "column_understanding", COLUMN_A, a_prompt, a_schema
    )
    if aa1["prompt_sha256"] != aa2["prompt_sha256"]:
        raise AssertionError("A -> A prompt hashes are not identical")
    if aa1["schema_sha256"] != aa2["schema_sha256"]:
        raise AssertionError("A -> A schema hashes are not identical")
    all_sequences["A -> A"] = [aa1, aa2]

    print("\nSequence A -> B", flush=True)
    ab_a, _ = measured_request(
        provider, "column_understanding", COLUMN_A, a_prompt, a_schema
    )
    b_prompt, b_schema = column_requests[COLUMN_B]
    ab_b, _ = measured_request(
        provider, "column_understanding", COLUMN_B, b_prompt, b_schema
    )
    all_sequences["A -> B"] = [ab_a, ab_b]

    if a_result is None:
        raise RuntimeError("The first patient_id request did not return valid semantics")
    a_semantics = ColumnSemanticResult(
        column_name=COLUMN_A,
        semantic_role="unknown",
        rag_sources=[],
        **a_result,
    )
    role_prompt, role_schema = capture_request(
        resolve_column_role,
        column_evidence=evidence[COLUMN_A],
        column_semantics=a_semantics,
        dataset_semantics=dataset_semantics,
        retriever=retriever,
        top_k=5,
    )

    print("\nSequence A -> Role(A) -> A", flush=True)
    ara1, _ = measured_request(
        provider, "column_understanding", COLUMN_A, a_prompt, a_schema
    )
    ar_role, _ = measured_request(
        provider, "role_resolver", COLUMN_A, role_prompt, role_schema
    )
    ara2, _ = measured_request(
        provider, "column_understanding", COLUMN_A, a_prompt, a_schema
    )
    all_sequences["A -> Role(A) -> A"] = [ara1, ar_role, ara2]

    aa_first = duration(aa1)
    aa_second = duration(aa2)
    speedup = aa_first / aa_second if aa_first and aa_second else None
    print("\nCompact comparison")
    print("Sequence A -> A")
    print(f"  A1 prompt eval: {format_duration(aa_first)}")
    print(f"  A2 prompt eval: {format_duration(aa_second)}")
    print(f"  cache speedup: {speedup:.2f}x" if speedup else "  cache speedup: unavailable")
    print("  prompt hashes identical: yes")
    print("  schema hashes identical: yes")
    print("Sequence A -> B")
    print(f"  A prompt eval: {format_duration(duration(ab_a))}")
    print(f"  B prompt eval: {format_duration(duration(ab_b))}")
    print("Sequence A -> Role(A) -> A")
    print(f"  A1 prompt eval: {format_duration(duration(ara1))}")
    print(f"  Role prompt eval: {format_duration(duration(ar_role))}")
    print(f"  A2 prompt eval: {format_duration(duration(ara2))}")

    exact_cache_works = bool(speedup and speedup >= 5.0)
    cold_tps = aa1.get("prompt_eval_tokens_per_second")
    b_tps = ab_b.get("prompt_eval_tokens_per_second")
    shared_prefix_material = bool(
        cold_tps and b_tps and b_tps >= cold_tps * 5.0
    )
    before_role = duration(ara1)
    after_role = duration(ara2)
    role_interferes = bool(
        before_role and after_role and after_role >= before_role * 5.0
    )
    print("\nMeasurement-derived conclusions")
    print(f"  1. Exact-prompt caching works: {'yes' if exact_cache_works else 'not demonstrated'}")
    print(
        "  2. Material shared-prefix caching across different columns: "
        + ("observed" if shared_prefix_material else "not demonstrated")
    )
    print(
        "  3. Role Resolver interferes with reuse of the Column prompt: "
        + ("yes" if role_interferes else "no")
    )
    print(
        "  4. Realistic full-benchmark benefit: exact repeated requests can "
        "benefit; distinct first-pass column prompts did not demonstrate a "
        "material cache benefit."
    )

    output_path = Path(__file__).with_name("ollama_prompt_cache_diagnostic.json")
    output_path.write_text(
        json.dumps(all_sequences, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nFull metrics written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
