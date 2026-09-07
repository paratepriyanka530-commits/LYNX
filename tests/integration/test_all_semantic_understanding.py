"""Run dataset- and column-level semantic understanding on 15 valid datasets.

Place at:
    tests/integration/test_all_semantic_understanding.py

Run from the LYNX project root:
    python tests/integration/test_all_semantic_understanding.py

Consolidated output:
    tests/integration/semantic_understanding_results.json
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.llm.ollama_provider import OllamaProvider
from intelligence.rag.retriever import RAGRetriever
from pipelines.pipeline1_cleaning.ingestion.dataset_loader import load_dataset
from pipelines.pipeline1_cleaning.profiling.dataset_profiler import profile_dataset
from pipelines.pipeline1_cleaning.semantics.evidence_builder import (
    build_dataset_evidence,
)
from pipelines.pipeline1_cleaning.semantics.dataset_understanding import (
    understand_dataset,
)
from pipelines.pipeline1_cleaning.semantics.column_understanding import (
    understand_column,
)


VALID_DATASET_DIR = PROJECT_ROOT / "tests" / "datasets" / "valid"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "tests"
    / "integration"
    / "semantic_understanding_results.json"
)
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
EXPECTED_DATASET_COUNT = 15
OLLAMA_MODEL = "qwen2.5:3b"
DATASET_RAG_TOP_K = 8
COLUMN_RAG_TOP_K = 6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_json_safe(value: Any) -> Any:
    """Convert pandas, NumPy, Pydantic, date, and Path values for JSON."""
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


def discover_datasets() -> list[Path]:
    if not VALID_DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {VALID_DATASET_DIR}"
        )

    datasets = sorted(
        (
            path
            for path in VALID_DATASET_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )

    if len(datasets) != EXPECTED_DATASET_COUNT:
        names = ", ".join(path.name for path in datasets) or "none"
        raise RuntimeError(
            f"Expected {EXPECTED_DATASET_COUNT} datasets but found "
            f"{len(datasets)}: {names}"
        )
    return datasets


def save_checkpoint(payload: dict[str, Any]) -> None:
    """Atomically preserve progress after each dataset/column inference."""
    temporary_file = OUTPUT_FILE.with_suffix(".json.tmp")
    temporary_file.write_text(
        json.dumps(make_json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_file.replace(OUTPUT_FILE)


def error_fields(error: Exception) -> dict[str, str]:
    return {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }


def main() -> int:
    datasets = discover_datasets()

    print(f"Found {len(datasets)} valid datasets.")
    print("Indexing semantic knowledge once...", flush=True)

    retriever = RAGRetriever(
        knowledge_directory=str(PROJECT_ROOT / "knowledge_base" / "semantic")
    )
    retriever.index_knowledge()
    indexed_chunks = retriever.vector_store.count()
    llm_provider = OllamaProvider(model=OLLAMA_MODEL)

    output: dict[str, Any] = {
        "test_suite": "LYNX dataset and all-column semantic evaluation",
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "run_status": "running",
        "configuration": {
            "expected_dataset_count": EXPECTED_DATASET_COUNT,
            "ollama_model": OLLAMA_MODEL,
            "dataset_rag_top_k": DATASET_RAG_TOP_K,
            "column_rag_top_k": COLUMN_RAG_TOP_K,
            "indexed_semantic_chunks": indexed_chunks,
        },
        "summary": {
            "datasets_discovered": len(datasets),
            "datasets_completed": 0,
            "dataset_semantics_passed": 0,
            "dataset_semantics_failed": 0,
            "columns_attempted": 0,
            "column_semantics_passed": 0,
            "column_semantics_failed": 0,
        },
        "datasets": [],
    }
    save_checkpoint(output)

    total_datasets = len(datasets)

    for dataset_number, dataset_path in enumerate(datasets, start=1):
        dataset_started = time.monotonic()
        print(
            f"\nDataset {dataset_number}/{total_datasets} running: "
            f"{dataset_path.name}",
            flush=True,
        )

        dataset_record: dict[str, Any] = {
            "dataset_number": dataset_number,
            "dataset_name": dataset_path.name,
            "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
            "dataset_class": (
                "messy"
                if dataset_path.name.startswith("dataset_")
                else "clean_control"
            ),
            "file_type": dataset_path.suffix.lower(),
            "started_at_utc": utc_now(),
            "completed_at_utc": None,
            "duration_seconds": None,
            "row_count": None,
            "column_count": None,
            "dataset_status": "running",
            "dataset_evidence": None,
            "dataset_semantics": None,
            "dataset_error": None,
            "column_summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
            },
            "columns": [],
        }
        output["datasets"].append(dataset_record)
        save_checkpoint(output)

        try:
            loaded = load_dataset(str(dataset_path))
            dataframe = loaded["dataframe"]
            dataset_record["row_count"] = int(dataframe.shape[0])
            dataset_record["column_count"] = int(dataframe.shape[1])

            dataset_profile = profile_dataset(dataframe)
            dataset_evidence = build_dataset_evidence(dataset_profile)
            dataset_record["dataset_evidence"] = make_json_safe(dataset_evidence)

            dataset_semantics = understand_dataset(
                evidence=dataset_evidence,
                retriever=retriever,
                llm_provider=llm_provider,
                top_k=DATASET_RAG_TOP_K,
            )
            dataset_record["dataset_semantics"] = model_to_dict(dataset_semantics)
            output["summary"]["dataset_semantics_passed"] += 1
            save_checkpoint(output)

        except Exception as error:
            dataset_record["dataset_status"] = "dataset_semantics_failed"
            dataset_record["dataset_error"] = error_fields(error)
            dataset_record["completed_at_utc"] = utc_now()
            dataset_record["duration_seconds"] = round(
                time.monotonic() - dataset_started, 3
            )
            output["summary"]["dataset_semantics_failed"] += 1
            output["summary"]["datasets_completed"] += 1
            save_checkpoint(output)
            print(
                f"Dataset {dataset_number}/{total_datasets} done: "
                "DATASET SEMANTICS FAILED",
                flush=True,
            )
            continue

        column_evidence_list = dataset_evidence.get("columns", [])
        dataset_record["column_summary"]["total"] = len(column_evidence_list)
        total_columns = len(column_evidence_list)

        for column_number, column_evidence in enumerate(
            column_evidence_list,
            start=1,
        ):
            column_started = time.monotonic()
            column_name = (
                column_evidence.get("column_name")
                or column_evidence.get("name")
                or f"unknown_column_{column_number}"
            )
            print(
                f"  Column {column_number}/{total_columns} running: "
                f"{column_name}",
                flush=True,
            )

            column_record: dict[str, Any] = {
                "column_number": column_number,
                "column_name": column_name,
                "status": "running",
                "duration_seconds": None,
                "column_evidence": make_json_safe(column_evidence),
                "column_semantics": None,
                "error": None,
            }
            dataset_record["columns"].append(column_record)
            output["summary"]["columns_attempted"] += 1

            try:
                column_semantics = understand_column(
                    column_evidence=column_evidence,
                    dataset_semantics=dataset_semantics,
                    retriever=retriever,
                    llm_provider=llm_provider,
                    top_k=COLUMN_RAG_TOP_K,
                )
                column_record["column_semantics"] = model_to_dict(
                    column_semantics
                )
                column_record["status"] = "passed"
                dataset_record["column_summary"]["passed"] += 1
                output["summary"]["column_semantics_passed"] += 1

            except Exception as error:
                column_record["status"] = "failed"
                column_record["error"] = error_fields(error)
                dataset_record["column_summary"]["failed"] += 1
                output["summary"]["column_semantics_failed"] += 1

            column_record["duration_seconds"] = round(
                time.monotonic() - column_started, 3
            )
            save_checkpoint(output)
            print(
                f"  Column {column_number}/{total_columns} done: "
                f"{column_record['status'].upper()}",
                flush=True,
            )

        dataset_record["dataset_status"] = (
            "completed"
            if dataset_record["column_summary"]["failed"] == 0
            else "completed_with_column_failures"
        )
        dataset_record["completed_at_utc"] = utc_now()
        dataset_record["duration_seconds"] = round(
            time.monotonic() - dataset_started, 3
        )
        output["summary"]["datasets_completed"] += 1
        save_checkpoint(output)

        print(
            f"Dataset {dataset_number}/{total_datasets} done: "
            f"{dataset_record['column_summary']['passed']}/"
            f"{total_columns} columns passed",
            flush=True,
        )

    output["completed_at_utc"] = utc_now()
    failures = (
        output["summary"]["dataset_semantics_failed"]
        + output["summary"]["column_semantics_failed"]
    )
    output["run_status"] = (
        "completed" if failures == 0 else "completed_with_failures"
    )
    save_checkpoint(output)

    print("\nAll semantic tests finished.")
    print(
        "Datasets: "
        f"{output['summary']['datasets_completed']}/{total_datasets} completed"
    )
    print(
        "Dataset semantics: "
        f"{output['summary']['dataset_semantics_passed']} passed, "
        f"{output['summary']['dataset_semantics_failed']} failed"
    )
    print(
        "Column semantics: "
        f"{output['summary']['column_semantics_passed']} passed, "
        f"{output['summary']['column_semantics_failed']} failed"
    )
    print(f"Results: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
