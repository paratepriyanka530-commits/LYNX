"""Resumably prepare privacy-cleared Pipeline 1 inputs for TabArena."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from benchmarks.tabarena.pipeline2_benchmark_runner import load_privacy_cleared_dataset
from benchmarks.tabarena.pipeline2_groundwork import DATASET_DIR, OUTPUT_DIR, load_manifest
from pipelines.pipeline1_cleaning.pipeline import run_pipeline1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def prepare_pipeline1_inputs(
    datasets: list[dict[str, Any]], output_root: Path,
    dataset_dir: Path = DATASET_DIR,
) -> list[dict[str, Any]]:
    """Run frozen Pipeline 1 per dataset, immediately persisting resumable status."""
    output_root = Path(output_root)
    records: list[dict[str, Any]] = []
    for index, dataset in enumerate(datasets, 1):
        dataset_id = str(dataset["task_id"])
        destination = output_root / dataset_id
        try:
            load_privacy_cleared_dataset(dataset, output_root)
            record = {"dataset_id": dataset_id, "status": "already_valid"}
            records.append(record)
            print(f"[SKIP] Pipeline 1 {index}/{len(datasets)} task_{dataset_id} already valid", flush=True)
            _atomic_json(output_root / "preparation_status.json", {
                "updated_at": _now(), "records": records,
            })
            continue
        except Exception:
            pass

        started = time.perf_counter()
        print(f"[START] Pipeline 1 {index}/{len(datasets)} task_{dataset_id}", flush=True)
        try:
            source = Path(dataset_dir) / dataset["csv_file"]
            run_pipeline1(source, destination)
            frame = load_privacy_cleared_dataset(dataset, output_root)
            record = {
                "dataset_id": dataset_id, "status": "complete",
                "row_count": len(frame),
                "duration_seconds": round(time.perf_counter() - started, 6),
            }
            print(
                f"[DONE] Pipeline 1 {index}/{len(datasets)} task_{dataset_id} "
                f"{record['duration_seconds']:.3f}s",
                flush=True,
            )
        except Exception as exc:
            record = {
                "dataset_id": dataset_id, "status": "error",
                "duration_seconds": round(time.perf_counter() - started, 6),
                "exception_type": type(exc).__name__, "exception_message": str(exc),
            }
            print(
                f"[ERROR] Pipeline 1 {index}/{len(datasets)} task_{dataset_id} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        records.append(record)
        _atomic_json(output_root / "preparation_status.json", {
            "updated_at": _now(), "records": records,
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--task-id", type=int, action="append")
    args = parser.parse_args()
    if args.all == bool(args.task_id):
        parser.error("Use exactly one of --all or --task-id")
    manifest = load_manifest(args.manifest_dir)
    selected = manifest if args.all else [
        item for item in manifest if int(item["task_id"]) in set(args.task_id)
    ]
    if not args.all and len(selected) != len(set(args.task_id)):
        parser.error("One or more --task-id values are absent from the manifest")
    records = prepare_pipeline1_inputs(selected, args.output_root)
    print(json.dumps(records, indent=2))
    return 0 if not any(record["status"] == "error" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
