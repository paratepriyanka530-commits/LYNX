"""Build and validate the TabArena target registry and imbalance audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


AUDIT_COLUMNS = [
    "dataset_id",
    "dataset_name",
    "task_type",
    "n_classes",
    "minority_class",
    "minority_class_count",
    "minority_class_ratio",
    "imbalance_category",
    "cb_chl_eligible",
]
SOURCE_DESCRIPTION = (
    "TabArena_51_Dataset_Target_Columns.pdf provided target mapping; OpenML "
    "suite 457 authoritative task metadata; local TabArena CSV target values"
)
RESOLUTION_SOURCE = "OpenML suite 457 dataset_manifest.json"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _task_type(authoritative_value: str) -> str:
    lowered = authoritative_value.casefold()
    if "classification" in lowered:
        return "classification"
    if "regression" in lowered:
        return "regression"
    raise ValueError(f"Unsupported authoritative task type: {authoritative_value!r}")


def _imbalance_category(ratio: float) -> str:
    if ratio < 0.01:
        return "EXTREME_IMBALANCE"
    if ratio < 0.05:
        return "SEVERE_IMBALANCE"
    if ratio < 0.20:
        return "MODERATE_IMBALANCE"
    return "NOT_IMBALANCED"


def _label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _unresolved_record(source: dict[str, Any], reason: str) -> dict[str, Any]:
    provided_target = source.get("target_column")
    return {
        "dataset_id": None,
        "dataset_name": source.get("dataset_name"),
        "dataset_file": source.get("dataset_file"),
        "target_column": None,
        "provided_target_column": provided_target,
        "authoritative_target_column": None,
        "benchmark_target_column": None,
        "target_resolution_status": "unresolved",
        "resolution_source": RESOLUTION_SOURCE,
        "task_type": None,
        "classification_type": None,
        "n_classes": None,
        "class_distribution": None,
        "minority_class_ratio": None,
        "target_row_count": None,
        "target_missing_count": None,
        "target_missing_pct": None,
        "imbalance_category": None,
        "cb_chl_eligible": False,
        "source": SOURCE_DESCRIPTION,
        "status": "unresolved",
        "unresolved_reason": reason,
    }


def _resolve_record(
    source: dict[str, Any],
    manifest_by_file: dict[str, dict[str, Any]],
    dataset_dir: Path,
) -> dict[str, Any]:
    dataset_file = source.get("dataset_file")
    if not isinstance(dataset_file, str) or not dataset_file:
        return _unresolved_record(source, "dataset_file is missing from the input registry")

    manifest = manifest_by_file.get(dataset_file)
    if manifest is None:
        return _unresolved_record(
            source,
            f"dataset_file {dataset_file!r} has no exact manifest match",
        )
    if source.get("dataset_name") != manifest.get("dataset_name"):
        return _unresolved_record(
            source,
            "dataset_name mismatch for exact dataset_file: "
            f"registry={source.get('dataset_name')!r}, "
            f"manifest={manifest.get('dataset_name')!r}",
        )

    provided_target = source.get("target_column")
    authoritative_target = manifest.get("target_column")
    if not isinstance(authoritative_target, str) or not authoritative_target:
        return _unresolved_record(
            source, "authoritative target_column is missing from the manifest"
        )
    benchmark_target = authoritative_target
    resolution_status = (
        "agreed"
        if provided_target == authoritative_target
        else "resolved_to_authoritative_target"
    )

    csv_path = dataset_dir / dataset_file
    if not csv_path.is_file():
        return _unresolved_record(source, f"local CSV does not exist: {csv_path}")
    try:
        columns = list(pd.read_csv(csv_path, nrows=0).columns)
    except Exception as exc:
        return _unresolved_record(
            source, f"CSV header read failed: {type(exc).__name__}: {exc}"
        )
    if benchmark_target not in columns:
        return _unresolved_record(
            source,
            f"benchmark_target_column {benchmark_target!r} is absent from the exact CSV header",
        )

    try:
        target = pd.read_csv(csv_path, usecols=[benchmark_target])[benchmark_target]
        task_type = _task_type(str(manifest.get("task_type")))
    except Exception as exc:
        return _unresolved_record(
            source, f"target read failed: {type(exc).__name__}: {exc}"
        )

    row_count = int(len(target))
    missing_count = int(target.isna().sum())
    missing_pct = (100.0 * missing_count / row_count) if row_count else 0.0
    record = {
        "dataset_id": int(manifest["task_id"]),
        "dataset_name": source["dataset_name"],
        "dataset_file": dataset_file,
        # target_column remains as a compatibility alias for benchmark execution.
        "target_column": benchmark_target,
        "provided_target_column": provided_target,
        "authoritative_target_column": authoritative_target,
        "benchmark_target_column": benchmark_target,
        "target_resolution_status": resolution_status,
        "resolution_source": RESOLUTION_SOURCE,
        "task_type": task_type,
        "classification_type": None,
        "n_classes": None,
        "class_distribution": None,
        "minority_class_ratio": None,
        "target_row_count": row_count,
        "target_missing_count": missing_count,
        "target_missing_pct": missing_pct,
        "imbalance_category": None,
        "cb_chl_eligible": False,
        "source": SOURCE_DESCRIPTION,
        "status": "resolved",
        "unresolved_reason": None,
    }
    if task_type == "regression":
        return record

    non_missing = target.dropna()
    counts = non_missing.value_counts(dropna=True, sort=False)
    n_classes = int(len(counts))
    if n_classes < 2:
        return _unresolved_record(
            source,
            f"authoritative classification target has only {n_classes} non-null class",
        )
    denominator = int(len(non_missing))
    distribution = {
        _label(value): {
            "count": int(count),
            "proportion": float(count / denominator),
        }
        for value, count in counts.items()
    }
    minority_value = counts.idxmin()
    minority_count = int(counts.loc[minority_value])
    minority_ratio = float(minority_count / denominator)
    record.update({
        "classification_type": "binary" if n_classes == 2 else "multiclass",
        "n_classes": n_classes,
        "class_distribution": distribution,
        "minority_class_ratio": minority_ratio,
        "imbalance_category": _imbalance_category(minority_ratio),
        "cb_chl_eligible": minority_ratio < 0.01,
    })
    return record


def _audit_record(record: dict[str, Any]) -> dict[str, Any]:
    distribution = record.get("class_distribution") or {}
    if distribution:
        minority_class, minority = min(
            distribution.items(), key=lambda item: item[1]["count"]
        )
        minority_count = minority["count"]
    else:
        minority_class = None
        minority_count = None
    return {
        "dataset_id": record.get("dataset_id"),
        "dataset_name": record.get("dataset_name"),
        "task_type": record.get("task_type"),
        "n_classes": record.get("n_classes"),
        "minority_class": minority_class,
        "minority_class_count": minority_count,
        "minority_class_ratio": record.get("minority_class_ratio"),
        "imbalance_category": record.get("imbalance_category"),
        "cb_chl_eligible": bool(record.get("cb_chl_eligible", False)),
    }


def _validate(records: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    resolved = [item for item in records if item["status"] == "resolved"]
    classification = [item for item in resolved if item["task_type"] == "classification"]
    regression = [item for item in resolved if item["task_type"] == "regression"]

    distribution_valid = all(
        item["class_distribution"]
        and sum(part["count"] for part in item["class_distribution"].values())
        == item["target_row_count"] - item["target_missing_count"]
        and abs(
            sum(part["proportion"] for part in item["class_distribution"].values())
            - 1.0
        )
        <= 1e-9
        for item in classification
    )
    regression_null = all(
        item[field] is None
        for item in regression
        for field in (
            "classification_type",
            "n_classes",
            "class_distribution",
            "minority_class_ratio",
            "imbalance_category",
        )
    ) and all(item["cb_chl_eligible"] is False for item in regression)
    classification_complete = all(
        item["minority_class_ratio"] is not None
        and item["n_classes"] == len(item["class_distribution"])
        and all(part["count"] > 0 for part in item["class_distribution"].values())
        for item in classification
    )
    checks = {
        "exact_dataset_count": len(records) == expected_count,
        "all_target_columns_resolved": len(resolved) == expected_count,
        "classification_distributions_valid": distribution_valid,
        "regression_classification_metadata_null": regression_null,
        "classification_minority_ratios_present": classification_complete,
        "missing_targets_excluded_from_classes": distribution_valid,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "total_datasets": len(records),
        "resolved_datasets": len(resolved),
        "unresolved_datasets": len(records) - len(resolved),
        "classification_count": len(classification),
        "regression_count": len(regression),
        "binary_classification_count": sum(
            item["classification_type"] == "binary" for item in classification
        ),
        "multiclass_classification_count": sum(
            item["classification_type"] == "multiclass" for item in classification
        ),
    }


def build_target_registry(
    source_registry_path: Path,
    manifest_path: Path,
    dataset_dir: Path,
    input_registry_path: Path,
    output_dir: Path,
    expected_count: int = 51,
) -> dict[str, Any]:
    source_payload = json.loads(source_registry_path.read_text(encoding="utf-8"))
    source_records = source_payload.get("datasets")
    if not isinstance(source_records, list):
        raise ValueError("Input target registry must contain a datasets list")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_records = manifest_payload.get("datasets")
    if not isinstance(manifest_records, list):
        raise ValueError("Dataset manifest must contain a datasets list")
    manifest_by_file = {item["csv_file"]: item for item in manifest_records}
    if len(manifest_by_file) != len(manifest_records):
        raise ValueError("Dataset manifest contains duplicate csv_file values")

    records = [
        _resolve_record(item, manifest_by_file, dataset_dir)
        for item in source_records
    ]
    validation = _validate(records, expected_count)
    registry_payload = {
        "schema_version": "2.1",
        "dataset_count": len(records),
        "validation": validation,
        "datasets": records,
    }
    _atomic_json(input_registry_path, registry_payload)
    _atomic_json(output_dir / "dataset_target_registry.json", registry_payload)

    audit_records = [_audit_record(item) for item in records]
    audit_payload = {
        "schema_version": "1.0",
        "dataset_count": len(audit_records),
        "validation": validation,
        "records": audit_records,
    }
    _atomic_json(output_dir / "imbalance_audit.json", audit_payload)
    with (output_dir / "imbalance_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(audit_records)
    return registry_payload


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "benchmarks" / "tabarena" / "pipeline2_artifacts" / "dataset_manifest.json",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=root / "benchmarks" / "tabarena" / "datasets",
    )
    parser.add_argument(
        "--input-registry",
        type=Path,
        default=root / "pipelines" / "pipeline2_prediction" / "inputs" / "dataset_target_registry.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "benchmarks" / "tabarena" / "pipeline2_artifacts",
    )
    args = parser.parse_args()
    payload = build_target_registry(
        args.source_registry.resolve(),
        args.manifest.resolve(),
        args.dataset_dir.resolve(),
        args.input_registry.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(payload["validation"], indent=2))
    unresolved = payload["validation"]["unresolved_datasets"]
    return 0 if payload["validation"]["passed"] and unresolved == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
