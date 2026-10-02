"""Resumable 40-query harness for language-understanding experiment V2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

from intelligence.llm.ollama_provider import OllamaProvider
from pipelines.pipeline3_results.nlp.experimental.schemas_v2 import (
    ExperimentResultV2,
)
from pipelines.pipeline3_results.nlp.experimental.semantic_interpreter_v2 import (
    PretrainedSemanticInterpreterV2,
    StructuralInterpretationError,
)


ROOT = Path(__file__).resolve().parents[4]

LANGUAGE_CORPUS_V2: tuple[str, ...] = (
    "Which category made us the most money?",
    "What sells the most?",
    "What is the least selling category?",
    "Which pizza size brings in the most revenue?",
    "Which category performs best?",
    "What is the most popular pizza?",
    "Which pizza gets ordered most often?",
    "What makes up most of our revenue?",
    "Which category has the highest value?",
    "Show revenue by pizza category.",
    "Show total sales by pizza category as a doughnut chart.",
    "How have sales changed over time?",
    "Which month was strongest?",
    "Which category did badly?",
    "What are the top five pizzas by revenue?",
    "Which pizza sold the fewest units?",
    "What stands out in this dataset?",
    "Is anything unusual?",
    "Show me where the money is coming from.",
    "Compare price and quantity.",
    "Which department has the highest employee compensation?",
    "Which hospital treated the most patients?",
    "Which account type has the highest average balance?",
    "Which sensor recorded the greatest temperature variation?",
    "Which student group performed best?",
    "Which machine had the most downtime?",
    "Which attack type occurred most frequently?",
    "Which survey response category was least common?",
    "Which product segment generated the highest margin?",
    "How did energy usage change throughout the year?",
    "Who is doing the worst?",
    "Where are we losing money?",
    "What changed the most?",
    "Which group is ahead?",
    "What is driving the numbers?",
    "Which one is underperforming?",
    "What happened last year?",
    "Are sales getting better?",
    "What should I be worried about?",
    "Which area contributes the most?",
)


def evaluate_query(
    query: str,
    interpreter: PretrainedSemanticInterpreterV2,
) -> ExperimentResultV2:
    started = time.perf_counter()
    try:
        output = interpreter.interpret(query)
        elapsed = time.perf_counter() - started
        return ExperimentResultV2(
            query=query,
            model_output=output.model_dump(mode="json"),
            schema_validation_passed=True,
            structural_validation_passed=True,
            latency_seconds=elapsed,
            provider_metadata=getattr(
                interpreter.provider, "last_response_metadata", None
            ),
            error=None,
        )
    except StructuralInterpretationError as exc:
        return ExperimentResultV2(
            query=query,
            model_output=exc.result.model_dump(mode="json"),
            schema_validation_passed=True,
            structural_validation_passed=False,
            latency_seconds=time.perf_counter() - started,
            provider_metadata=getattr(
                interpreter.provider, "last_response_metadata", None
            ),
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:
        return ExperimentResultV2(
            query=query,
            model_output=_raw_output_from_error(exc),
            schema_validation_passed=False,
            structural_validation_passed=False,
            latency_seconds=time.perf_counter() - started,
            provider_metadata=getattr(
                interpreter.provider, "last_response_metadata", None
            ),
            error=f"{type(exc).__name__}: {exc}",
        )


def _raw_output_from_error(exc: Exception) -> dict[str, Any] | None:
    marker = "Raw response:\n"
    message = str(exc)
    if marker not in message:
        return None
    raw = message.split(marker, 1)[1].strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _load_existing_result(item: dict[str, Any]) -> ExperimentResultV2:
    """Upgrade the first-pass checkpoint without rerunning captured responses."""
    upgraded = dict(item)
    if "schema_validation_passed" not in upgraded:
        error = upgraded.get("error")
        raw = None
        if upgraded.get("model_output") is None and isinstance(error, str):
            raw = _raw_output_from_error(RuntimeError(error))
        if raw is not None:
            upgraded["model_output"] = raw
            upgraded["schema_validation_passed"] = False
        else:
            upgraded["schema_validation_passed"] = bool(
                upgraded.get("structural_validation_passed")
            )
    return ExperimentResultV2.model_validate(upgraded)


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def run_corpus(
    *,
    provider: OllamaProvider,
    output_path: Path,
    queries: Iterable[str] = LANGUAGE_CORPUS_V2,
    resume: bool = True,
) -> list[ExperimentResultV2]:
    interpreter = PretrainedSemanticInterpreterV2(provider)
    results: list[ExperimentResultV2] = []
    if resume and output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        results = [_load_existing_result(item) for item in payload["results"]]
    completed = {item.query for item in results if item.model_output is not None}
    selected = list(queries)
    for index, query in enumerate(selected, start=1):
        if query in completed:
            print(f"[SKIP] {index}/{len(selected)} {query}", flush=True)
            continue
        print(f"[START] {index}/{len(selected)} {query}", flush=True)
        result = evaluate_query(query, interpreter)
        results = [item for item in results if item.query != query]
        results.append(result)
        _atomic_write(output_path, {
            "experiment": "pipeline3_pretrained_language_interpreter_v2",
            "model": provider.model,
            "scope": "language_understanding_only_no_grounding",
            "correctness_labels": "manual_unscored",
            "results": [item.model_dump(mode="json") for item in results],
        })
        marker = "DONE" if result.error is None else "ERROR"
        print(f"[{marker}] {index}/{len(selected)} {result.latency_seconds:.2f}s", flush=True)
    order = {query: index for index, query in enumerate(selected)}
    results.sort(key=lambda item: order.get(item.query, len(order)))
    _atomic_write(output_path, {
        "experiment": "pipeline3_pretrained_language_interpreter_v2",
        "model": provider.model,
        "scope": "language_understanding_only_no_grounding",
        "correctness_labels": "manual_unscored",
        "results": [item.model_dump(mode="json") for item in results],
    })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tests" / "manual" / "pipeline3_nlp_experiment_v2.json",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    queries = (
        LANGUAGE_CORPUS_V2[: args.limit]
        if args.limit is not None
        else LANGUAGE_CORPUS_V2
    )
    results = run_corpus(
        provider=OllamaProvider(model=args.model, base_url=args.base_url),
        output_path=args.output,
        queries=queries,
        resume=not args.no_resume,
    )
    failures = sum(item.error is not None for item in results)
    print(f"Completed {len(results)} comparison(s); errors={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LANGUAGE_CORPUS_V2", "evaluate_query", "run_corpus"]
