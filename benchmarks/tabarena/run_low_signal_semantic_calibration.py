"""Column-only semantic calibration diagnostic for opaque MIC column codes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tabarena.pipeline2_groundwork import GroundworkOllamaProvider
from intelligence.rag.retriever import RAGRetriever
from pipelines.pipeline1_cleaning.profiling.dataset_profiler import profile_dataset
from pipelines.pipeline1_cleaning.semantics.batched_analyzer import (
    BatchSemanticConfig,
    build_batch_prompt,
    build_batch_rag_query,
    build_compact_column_evidence,
    create_batches,
    validate_batch_coverage,
)
from pipelines.pipeline1_cleaning.semantics.batched_schema import (
    BatchedColumnInference,
    BatchedDatasetContext,
)


DEFAULT_DATASET = ROOT / "benchmarks" / "tabarena" / "datasets" / "MIC.csv"
DEFAULT_OUTPUT = (
    ROOT / "benchmarks" / "tabarena" / "pipeline2_artifacts"
    / "low_signal_mic_qwen25_7b.json"
)
HIGH_CONFIDENCE = 0.80
TRANSPARENT_EXACT_NAMES = {"age", "sex"}
TRANSPARENT_TOKENS = {"age", "sex", "blood", "time", "date", "year", "month", "day"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def name_is_opaque(name: str) -> bool:
    lowered = name.casefold()
    if lowered in TRANSPARENT_EXACT_NAMES:
        return False
    tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
    if any(token in TRANSPARENT_TOKENS for token in tokens):
        return False
    return bool(re.search(r"\d|_", name)) or (name.isupper() and len(name) <= 12)


def statistically_ambiguous_numeric_code(series: pd.Series) -> tuple[bool, dict[str, Any]]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    non_null = series.dropna()
    conversion_ratio = float(len(numeric) / len(non_null)) if len(non_null) else 0.0
    unique = numeric.nunique(dropna=True)
    integer_like = bool(len(numeric)) and all(
        math.isclose(float(value), round(float(value)), abs_tol=1e-9)
        for value in numeric.unique()
    )
    ambiguous = conversion_ratio == 1.0 and integer_like and unique <= 10
    return ambiguous, {
        "numeric_conversion_ratio": conversion_ratio,
        "observed_unique_count": int(unique),
        "all_observed_values_integer_like": integer_like,
        "ambiguity_basis": (
            "Low-cardinality integer storage is compatible with boolean, categorical, "
            "ordinal, or nominal-code meanings."
        ) if ambiguous else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "column_name", "semantic_type", "confidence", "rationale",
        "opaque_name", "statistically_ambiguous", "fabricated_confidence_candidate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataframe = pd.read_csv(args.dataset, low_memory=False)
    deterministic_profile = profile_dataset(dataframe)
    compact = build_compact_column_evidence(deterministic_profile, sample_cap=5)
    context = BatchedDatasetContext(
        row_entity=None,
        row_entity_confidence=0.0,
        probable_domain=None,
        domain_confidence=None,
        dataset_description=None,
    )
    retriever = RAGRetriever(knowledge_directory=str(ROOT / "knowledge_base"))
    retriever.index_knowledge()
    provider = GroundworkOllamaProvider(args.model, args.timeout)
    config = BatchSemanticConfig(max_columns_per_batch=args.batch_size, retry_count=args.retries)
    batches = create_batches(compact, config.max_columns_per_batch)
    results: dict[str, dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []
    batch_failures: list[dict[str, Any]] = []
    started = time.perf_counter()

    document: dict[str, Any] = {
        "diagnostic": "low_signal_semantic_calibration",
        "dataset": str(args.dataset.resolve()),
        "model": args.model,
        "started_at": utc_now(),
        "configuration": {
            "column_level_only": True,
            "dataset_context_llm_call": False,
            "batch_size": args.batch_size,
            "retry_count": args.retries,
            "timeout_seconds": args.timeout,
            "high_confidence_threshold": HIGH_CONFIDENCE,
            "opaque_name_heuristic": (
                "Not AGE/SEX; no transparent token (age, sex, blood, time, date, year, "
                "month, day); and contains '_' or a digit, or is all-uppercase <=12 chars."
            ),
            "statistical_ambiguity_heuristic": (
                "All non-null values numeric and integer-like with <=10 distinct values."
            ),
        },
        "input_shape": {"rows": len(dataframe), "columns": len(dataframe.columns)},
        "calls": calls,
        "batch_failures": batch_failures,
        "columns": [],
        "summary": None,
    }

    for batch_index, batch in enumerate(batches, start=1):
        expected = [item["column_name"] for item in batch]
        retrieved = retriever.retrieve(
            build_batch_rag_query(batch, context),
            top_k=config.batch_rag_top_k,
            categories=["semantic", "semantic_concept"],
        )
        deduplicated, seen = [], set()
        for item in retrieved:
            if item.get("chunk_id") not in seen:
                seen.add(item.get("chunk_id"))
                deduplicated.append(item)
        prompt = build_batch_prompt(batch, context, deduplicated)
        resolved = None
        last_error = None
        for attempt in range(config.retry_count + 1):
            call_started = time.perf_counter()
            try:
                raw = provider.generate_structured(prompt, BatchedColumnInference)
                parsed = BatchedColumnInference(**raw)
                resolved = validate_batch_coverage(expected, parsed.columns)
                calls.append({
                    "batch_index": batch_index, "attempt": attempt + 1,
                    "columns": expected, "status": "success",
                    "wall_seconds": round(time.perf_counter() - call_started, 6),
                    "ollama": provider.last_response_metadata,
                })
                break
            except Exception as exc:
                last_error = exc
                calls.append({
                    "batch_index": batch_index, "attempt": attempt + 1,
                    "columns": expected, "status": "error",
                    "wall_seconds": round(time.perf_counter() - call_started, 6),
                    "error": f"{type(exc).__name__}: {exc}",
                    "ollama": provider.last_response_metadata,
                })
        if resolved is None:
            batch_failures.append({
                "batch_index": batch_index, "columns": expected,
                "error": f"{type(last_error).__name__}: {last_error}",
            })
        else:
            for name in expected:
                results[name] = resolved[name].model_dump(mode="json")
        document["columns_completed"] = len(results)
        document["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        atomic_json(args.output, document)
        print(
            f"batch {batch_index}/{len(batches)}: "
            f"{'success' if resolved is not None else 'failed'} "
            f"({len(results)}/{len(dataframe.columns)} columns)",
            flush=True,
        )

    rows = []
    for name in map(str, dataframe.columns):
        semantic = results.get(name)
        opaque = name_is_opaque(name)
        ambiguous, ambiguity_evidence = statistically_ambiguous_numeric_code(dataframe[name])
        specific_high = bool(
            semantic
            and semantic["semantic_type"] != "unknown"
            and float(semantic["confidence"]) >= HIGH_CONFIDENCE
        )
        rows.append({
            "column_name": name,
            "semantic_type": semantic["semantic_type"] if semantic else "unresolved_batch_failure",
            "confidence": semantic["confidence"] if semantic else None,
            "rationale": semantic["reasoning_summary"] if semantic else "Semantic batch failed.",
            "value_function": semantic["value_function"] if semantic else None,
            "numeric_meaning": semantic["numeric_meaning"] if semantic else None,
            "opaque_name": opaque,
            "statistically_ambiguous": ambiguous,
            "ambiguity_evidence": ambiguity_evidence,
            "specific_high_confidence": specific_high,
            "fabricated_confidence_candidate": specific_high and opaque and ambiguous,
        })
    candidates = [row for row in rows if row["fabricated_confidence_candidate"]]
    document.update({
        "completed_at": utc_now(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "columns": rows,
        "summary": {
            "total_columns": len(rows),
            "resolved_columns": len(results),
            "batch_failure_count": len(batch_failures),
            "specific_high_confidence_count": sum(row["specific_high_confidence"] for row in rows),
            "opaque_and_statistically_ambiguous_count": sum(
                row["opaque_name"] and row["statistically_ambiguous"] for row in rows
            ),
            "fabricated_confidence_candidate_count": len(candidates),
            "fabricated_confidence_candidate_columns": [row["column_name"] for row in candidates],
        },
    })
    atomic_json(args.output, document)
    write_csv(args.output.with_suffix(".csv"), rows)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=480)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result["summary"], indent=2))
    return 0 if not result["batch_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
