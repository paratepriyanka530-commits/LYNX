"""Controlled synthetic feature-capacity probe for the merged XTab wrapper."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_FEATURE_COUNTS = (20, 50, 100, 150, 200, 300, 500)


def _memory_snapshot():
    try:
        import psutil

        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset")) if hasattr(info, "peak_wset") else None
    except (ImportError, OSError):
        return None


def _gpu_peak(reset=False):
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        if reset:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            return None
        return int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        return None


def probe_one(feature_count: int, rows: int, model_package=None) -> dict[str, Any]:
    if model_package is None:
        from pipelines.pipeline2_prediction.models import xtab as model_package

    rng = np.random.default_rng(42 + feature_count)
    features = pd.DataFrame(
        rng.normal(size=(rows, feature_count)),
        columns=[f"feature_{index:04d}" for index in range(feature_count)],
    )
    target = pd.Series(np.arange(rows) % 2)
    result = {
        "feature_count": feature_count,
        "rows": rows,
        "device": None,
        "success": False,
        "exception_type": None,
        "exception_message": None,
        "runtime_seconds": None,
        "peak_gpu_memory_bytes": None,
        "peak_process_memory_bytes": None,
        "prediction_success": False,
        "output_shape": None,
    }
    _gpu_peak(reset=True)
    started = time.perf_counter()
    try:
        model = model_package.build_model(
            "classification", random_state=42, epochs=1, patience=1
        )
        model_package.train_model(model, features, target)
        predictions = np.asarray(
            model_package.predict_model(model, features.iloc[:8])
        ).reshape(-1)
        prediction_success = bool(
            predictions.shape == (8,) and pd.notna(predictions).all()
        )
        result.update({
            "device": str(model.trainer.device),
            "success": True,
            "prediction_success": prediction_success,
            "output_shape": list(predictions.shape),
        })
    except Exception as exc:
        result.update({
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        })
    finally:
        result["runtime_seconds"] = time.perf_counter() - started
        result["peak_gpu_memory_bytes"] = _gpu_peak()
        result["peak_process_memory_bytes"] = _memory_snapshot()
    return result


def run_capacity_probe(
    output_dir: Path,
    feature_counts=DEFAULT_FEATURE_COUNTS,
    rows: int = 96,
    max_runtime_seconds: float = 300.0,
    model_package=None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    stop_reason = None
    for feature_count in feature_counts:
        record = probe_one(feature_count, rows, model_package=model_package)
        records.append(record)
        message = (record.get("exception_message") or "").casefold()
        if record["exception_type"] in {"MemoryError", "OutOfMemoryError"} or "out of memory" in message:
            stop_reason = f"memory boundary observed at {feature_count} features"
            break
        if record["runtime_seconds"] > max_runtime_seconds:
            stop_reason = f"runtime threshold exceeded at {feature_count} features"
            break

    successful = [item["feature_count"] for item in records if item["success"]]
    payload = {
        "diagnostic": "xtab_feature_capacity",
        "classification": "EMPIRICAL_SAFE_RANGE",
        "rows_per_probe": rows,
        "tested_feature_counts": [item["feature_count"] for item in records],
        "highest_successful_feature_count": max(successful) if successful else None,
        "hard_maximum_features": None,
        "memory_measurement_notes": (
            "CUDA max_memory_allocated is reset per probe; process memory is "
            "the Windows process-lifetime peak working set observed by psutil."
        ),
        "stop_reason": stop_reason,
        "records": records,
    }
    (output_dir / "xtab_feature_capacity.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "xtab_feature_capacity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=96)
    parser.add_argument(
        "--features", type=int, nargs="+", default=list(DEFAULT_FEATURE_COUNTS)
    )
    parser.add_argument("--max-runtime-seconds", type=float, default=300.0)
    args = parser.parse_args()
    payload = run_capacity_probe(
        args.output_dir,
        feature_counts=args.features,
        rows=args.rows,
        max_runtime_seconds=args.max_runtime_seconds,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["highest_successful_feature_count"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
