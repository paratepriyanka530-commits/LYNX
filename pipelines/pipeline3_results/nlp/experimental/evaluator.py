"""Manual comparison harness for deterministic and pretrained-first NLP.

Run from the repository root with::

    python -m pipelines.pipeline3_results.nlp.experimental.evaluator \
        --model qwen2.5:7b --output tests/manual/pipeline3_nlp_experiment.json

The output is resumable by query and is rewritten atomically after each case.
No correctness score is calculated; the eight evaluation labels remain null
until a human annotates them.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

import pandas as pd

from intelligence.llm.ollama_provider import OllamaProvider
from pipelines.pipeline3_results.nlp.experimental.grounding import (
    ground_interpretation,
)
from pipelines.pipeline3_results.nlp.experimental.schemas import (
    ExperimentalComparison,
)
from pipelines.pipeline3_results.nlp.experimental.semantic_interpreter import (
    PretrainedSemanticInterpreter,
)
from pipelines.pipeline3_results.nlp.orchestrator import interpret_query


ROOT = Path(__file__).resolve().parents[4]
DATASET_DIRECTORY = ROOT / "tests" / "datasets" / "valid"
REPORT_DIRECTORY = ROOT / "tests" / "pipeline1_multidomain" / "results_v001_baseline"
PIZZA_REPORT = (
    ROOT / "tests" / "integration" / "pipeline1_architecture_b_pizza_output_v3"
    / "cleaning_report.json"
)


@dataclass(frozen=True)
class CorpusCase:
    query: str
    dataset_key: str


TEST_CORPUS: tuple[CorpusCase, ...] = (
    CorpusCase("Which category made us the most money?", "pizza"),
    CorpusCase("What sells the most?", "pizza"),
    CorpusCase("What is the least selling category?", "pizza"),
    CorpusCase("Which pizza size brings in the most revenue?", "pizza"),
    CorpusCase("Which category performs best?", "pizza"),
    CorpusCase("What is the most popular pizza?", "pizza"),
    CorpusCase("Which pizza gets ordered most often?", "pizza"),
    CorpusCase("What makes up most of our revenue?", "pizza"),
    CorpusCase("Which category has the highest value?", "pizza"),
    CorpusCase("Show revenue by pizza category.", "pizza"),
    CorpusCase("Show total sales by pizza category as a doughnut chart.", "pizza"),
    CorpusCase("How have sales changed over time?", "pizza"),
    CorpusCase("Which month was strongest?", "pizza"),
    CorpusCase("Which category did badly?", "pizza"),
    CorpusCase("What are the top five pizzas by revenue?", "pizza"),
    CorpusCase("Which pizza sold the fewest units?", "pizza"),
    CorpusCase("What stands out in this dataset?", "pizza"),
    CorpusCase("Is anything unusual?", "pizza"),
    CorpusCase("Show me where the money is coming from.", "pizza"),
    CorpusCase("Compare price and quantity.", "pizza"),
    CorpusCase("Which department has the highest employee compensation?", "hr"),
    CorpusCase("Which hospital treated the most patients?", "healthcare"),
    CorpusCase("Which account type has the highest average balance?", "banking"),
    CorpusCase("Which sensor recorded the greatest temperature variation?", "iot"),
    CorpusCase("Which student group performed best?", "education"),
    CorpusCase("Which machine had the most downtime?", "manufacturing"),
    CorpusCase("Which attack type occurred most frequently?", "cybersecurity"),
    CorpusCase("Which survey response category was least common?", "survey"),
    CorpusCase("Which product segment generated the highest margin?", "retail"),
    CorpusCase("How did energy usage change throughout the year?", "iot"),
)


_DATASETS: dict[str, tuple[str, str | None]] = {
    "pizza": ("pizza_sales.csv", None),
    "retail": ("dataset_01_retail_transactions_raw.csv", "dataset_01_retail_transactions"),
    "hr": ("dataset_02_hr_employees_raw.csv", "dataset_02_hr_employees"),
    "healthcare": ("dataset_03_healthcare_raw.csv", "dataset_03_healthcare"),
    "banking": ("dataset_04_banking_finance_raw.csv", "dataset_04_banking_finance"),
    "education": ("dataset_05_education_raw.csv", "dataset_05_education"),
    "cybersecurity": ("dataset_06_cybersecurity_raw.csv", "dataset_06_cybersecurity"),
    "iot": ("dataset_07_iot_sensor_raw.csv", "dataset_07_iot_sensor"),
    "manufacturing": ("dataset_08_manufacturing_raw.csv", "dataset_08_manufacturing"),
    "survey": ("dataset_09_survey_data_raw.csv", "dataset_09_survey_data"),
}


@lru_cache(maxsize=None)
def load_dataset_context(dataset_key: str) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    filename, report_folder = _DATASETS[dataset_key]
    dataframe = pd.read_csv(DATASET_DIRECTORY / filename)
    report_path = (
        PIZZA_REPORT
        if dataset_key == "pizza"
        else REPORT_DIRECTORY / str(report_folder) / "cleaning_report.json"
    )
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else None
    )
    return dataframe, report


@lru_cache(maxsize=None)
def load_dataset_classification(dataset_key: str):
    from pipelines.pipeline3_results.visualization.column_classifier import (
        classify_dataframe,
    )

    dataframe, report = load_dataset_context(dataset_key)
    return classify_dataframe(
        dataframe,
        pipeline1_report=report,
        dataset_name=dataset_key,
    )


def evaluate_case(
    case: CorpusCase,
    interpreter: PretrainedSemanticInterpreter,
) -> ExperimentalComparison:
    stage_started = time.perf_counter()
    dataframe, report = load_dataset_context(case.dataset_key)
    classification = load_dataset_classification(case.dataset_key)
    print(
        f"[LOCAL] dataset context/classification {time.perf_counter() - stage_started:.2f}s",
        flush=True,
    )
    stage_started = time.perf_counter()
    deterministic = interpret_query(
        case.query,
        dataframe,
        pipeline1_report=report,
        classification=classification,
    ).model_dump(mode="json")
    print(
        f"[LOCAL] deterministic interpretation {time.perf_counter() - stage_started:.2f}s",
        flush=True,
    )
    started = time.perf_counter()
    try:
        pretrained = interpreter.interpret(case.query)
        elapsed = time.perf_counter() - started
        print(f"[LLM] pretrained interpretation {elapsed:.2f}s", flush=True)
        stage_started = time.perf_counter()
        grounded = ground_interpretation(
            pretrained,
            dataframe,
            pipeline1_report=report,
            classification=classification,
        )
        print(
            f"[LOCAL] deterministic grounding {time.perf_counter() - stage_started:.2f}s",
            flush=True,
        )
        metadata = getattr(interpreter.provider, "last_response_metadata", None)
        return ExperimentalComparison(
            query=case.query,
            dataset_key=case.dataset_key,
            deterministic=deterministic,
            pretrained=pretrained,
            grounded_pretrained=grounded,
            latency_seconds=elapsed,
            provider_metadata=metadata,
        )
    except Exception as exc:  # preserve the rest of a long manual experiment
        return ExperimentalComparison(
            query=case.query,
            dataset_key=case.dataset_key,
            deterministic=deterministic,
            latency_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
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
    cases: Iterable[CorpusCase] = TEST_CORPUS,
    resume: bool = True,
) -> list[ExperimentalComparison]:
    interpreter = PretrainedSemanticInterpreter(provider)
    results: list[ExperimentalComparison] = []
    if resume and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        results = [ExperimentalComparison.model_validate(item) for item in existing["results"]]
    completed = {(item.query, item.dataset_key) for item in results if item.error is None}
    selected = list(cases)
    for index, case in enumerate(selected, start=1):
        if (case.query, case.dataset_key) in completed:
            print(f"[SKIP] {index}/{len(selected)} {case.dataset_key}: {case.query}", flush=True)
            continue
        print(f"[START] {index}/{len(selected)} {case.dataset_key}: {case.query}", flush=True)
        result = evaluate_case(case, interpreter)
        results = [
            item for item in results
            if (item.query, item.dataset_key) != (case.query, case.dataset_key)
        ]
        results.append(result)
        payload = {
            "experiment": "pipeline3_pretrained_primary_semantic_interpreter_v1",
            "model": provider.model,
            "correctness_labels": "manual_unscored",
            "results": [item.model_dump(mode="json") for item in results],
        }
        _atomic_write(output_path, payload)
        marker = "ERROR" if result.error else "DONE"
        print(
            f"[{marker}] {index}/{len(selected)} {result.latency_seconds:.2f}s",
            flush=True,
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tests" / "manual" / "pipeline3_nlp_experiment.json",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cases = TEST_CORPUS[: args.limit] if args.limit is not None else TEST_CORPUS
    results = run_corpus(
        provider=OllamaProvider(model=args.model, base_url=args.base_url),
        output_path=args.output,
        cases=cases,
        resume=not args.no_resume,
    )
    failures = sum(item.error is not None for item in results)
    print(f"Completed {len(results)} comparison(s); errors={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CorpusCase",
    "TEST_CORPUS",
    "evaluate_case",
    "load_dataset_context",
    "run_corpus",
]
