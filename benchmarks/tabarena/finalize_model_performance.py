"""Build the authoritative selector-supervision artifact from benchmark rows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tabarena.pipeline2_groundwork import EVALUATION_MODEL_IDS  # noqa: E402
from pipelines.pipeline2_prediction.build_selector_inputs import (  # noqa: E402
    performance_template,
    validate_performance_record,
)


DEFAULT_OUTPUT = (
    ROOT / "pipelines" / "pipeline2_prediction" / "training_data"
    / "model_performance.json"
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _performance_record(row: dict[str, Any]) -> dict[str, Any]:
    metric_name = row.get("metric_name")
    metric_value = row.get("metric_value")
    direction = None
    if metric_name is not None:
        direction = "minimize" if metric_name == "rmse" else "maximize"
    return {
        "dataset_id": int(row["dataset_id"]),
        "model_id": row["model_id"],
        "base_model_id": row["base_model_id"],
        "execution_strategy": row["execution_strategy"],
        "run_id": row["run_id"],
        "task_type": row["task_type"],
        "feasible": row["feasible"],
        "status": row["status"],
        "split": row["split"],
        "infeasible_reason": row.get("infeasible_reason"),
        "hyperparameters": row.get("hyperparams") or {},
        "predictive_metrics": row.get("predictive_metrics"),
        "calibration_metrics": row.get("calibration_metrics"),
        "probability_metadata": row.get("probability_metadata"),
        "primary_metric": {
            "name": metric_name,
            "value": metric_value,
            "direction": direction,
        },
        "runtime": {
            "train_seconds": row.get("train_runtime_seconds"),
            "inference_seconds": row.get("inference_runtime_seconds"),
        },
        "resource_usage": row.get("resource_usage"),
        "model_specific_metrics": {
            "random_forest_oob_score": row.get("oob_score"),
            "random_forest_oob_diagnostic_reason": row.get("oob_diagnostic_reason"),
            "baseline_relative_score": row.get("baseline_relative_score"),
        },
        "protocol_provenance": {
            "source": row.get("source"),
            "target_column": row.get("target_column"),
            "privacy_gate": "validated Pipeline 1 cleaned and privacy-cleared output",
            "preprocessing": "fit on training split only; test transformed without refit",
            "selector_availability": "TRAINING_OUTCOME_ONLY",
        },
        "failure": row.get("failure"),
    }


def finalize(
    checkpoint_path: Path,
    output_path: Path = DEFAULT_OUTPUT,
    expected_count: int = 510,
) -> dict[str, Any]:
    rows = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Benchmark checkpoint must be a JSON list")
    records = [_performance_record(row) for row in rows]
    keys = [
        (
            record["dataset_id"], record["model_id"],
            record["execution_strategy"], record["run_id"],
        )
        for record in records
    ]
    if len(records) != expected_count or len(set(keys)) != expected_count:
        raise ValueError(
            f"Expected {expected_count} unique performance rows, found "
            f"{len(records)} rows and {len(set(keys))} unique keys"
        )
    for record in records:
        validate_performance_record(record)
    if expected_count == 510:
        dataset_ids = {record["dataset_id"] for record in records}
        models_by_dataset = {
            dataset_id: {
                record["base_model_id"] for record in records
                if record["dataset_id"] == dataset_id
                and record["execution_strategy"] == "standard"
            }
            for dataset_id in dataset_ids
        }
        if len(dataset_ids) != 51 or any(
            models != set(EVALUATION_MODEL_IDS) for models in models_by_dataset.values()
        ):
            raise ValueError("Performance rows do not form the canonical 51 x 10 matrix")
    payload = performance_template()
    payload.update({
        "description": "Observed benchmark supervision; never an inference-time selector feature.",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "record_count": len(records),
        "records": records,
        "validation": {
            "passed": True,
            "unique_run_identity_count": len(set(keys)),
            "success_count": sum(record["status"] == "success" for record in records),
            "infeasible_skip_count": sum(
                record["status"] == "infeasible_skip" for record in records
            ),
            "failure_count": sum(record["status"] == "failure" for record in records),
            "failed_or_infeasible_encoded_as_performance_zero": False,
            "selector_availability": "TRAINING_OUTCOME_ONLY",
        },
    })
    _atomic_json(Path(output_path), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-count", type=int, default=510)
    args = parser.parse_args()
    payload = finalize(args.checkpoint, args.output, args.expected_count)
    print(json.dumps(payload["validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
