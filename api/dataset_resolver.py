"""Dataset resolution logic for LYNX visualization API endpoints.

Resolves a caller-supplied dataset_id to an already-cleaned CSV and optional
Pipeline 1 cleaning report on disk without running Pipeline 1.

Supported dataset_id conventions:
1. TabArena benchmark tasks:
   - Identifiers: "363612", "task_363612", "tabarena:363612", "tabarena/363612", etc.
   - Maps to: benchmarks/tabarena/pipeline1_outputs/{task_id}/cleaned_dataset.csv
   - Report: benchmarks/tabarena/pipeline1_outputs/{task_id}/cleaning_report.json

2. Domain datasets:
   - Exact filenames (with or without .csv): "pizza_sales", "dataset_01_retail_transactions_raw", etc.
   - Convenient aliases: "pizza", "retail", "iot", "survey", "ecommerce", etc.
   - Maps to: storage/domain_datasets/{filename}.csv  (primary)
              tests/datasets/valid/{filename}.csv      (fallback for tests)

3. Application Run UUIDs:
   - Identifiers: Valid UUID string (e.g. from POST /api/runs)
   - Maps to: {runtime_root}/{run_id}/pipeline1/cleaned_dataset.csv
   - Report: {runtime_root}/{run_id}/pipeline1/cleaning_report.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

import pandas as pd

from api.run_service import ServiceError

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Aliases for domain valid datasets
_DOMAIN_ALIASES: dict[str, str] = {
    "pizza": "pizza_sales.csv",
    "pizza_sales": "pizza_sales.csv",
    "retail": "dataset_01_retail_transactions_raw.csv",
    "dataset_01": "dataset_01_retail_transactions_raw.csv",
    "hr": "dataset_02_hr_employees_raw.csv",
    "dataset_02": "dataset_02_hr_employees_raw.csv",
    "healthcare": "dataset_03_healthcare_raw.csv",
    "dataset_03": "dataset_03_healthcare_raw.csv",
    "finance": "dataset_04_banking_finance_raw.csv",
    "banking": "dataset_04_banking_finance_raw.csv",
    "dataset_04": "dataset_04_banking_finance_raw.csv",
    "education": "dataset_05_education_raw.csv",
    "dataset_05": "dataset_05_education_raw.csv",
    "cybersecurity": "dataset_06_cybersecurity_raw.csv",
    "security": "dataset_06_cybersecurity_raw.csv",
    "dataset_06": "dataset_06_cybersecurity_raw.csv",
    "iot": "dataset_07_iot_sensor_raw.csv",
    "iot_sensor": "dataset_07_iot_sensor_raw.csv",
    "dataset_07": "dataset_07_iot_sensor_raw.csv",
    "manufacturing": "dataset_08_manufacturing_raw.csv",
    "dataset_08": "dataset_08_manufacturing_raw.csv",
    "survey": "dataset_09_survey_data_raw.csv",
    "dataset_09": "dataset_09_survey_data_raw.csv",
    "ecommerce": "dataset_10_ecommerce_raw.csv",
    "dataset_10": "dataset_10_ecommerce_raw.csv",
    "scientific": "dataset_11_scientific_research_raw.csv",
    "dataset_11": "dataset_11_scientific_research_raw.csv",
    "logs": "dataset_12_log_event_data_raw.csv",
    "dataset_12": "dataset_12_log_event_data_raw.csv",
    "business": "dataset_13_generic_business_data_raw.csv",
    "dataset_13": "dataset_13_generic_business_data_raw.csv",
}


@dataclass(frozen=True)
class ResolvedDataset:
    dataframe: pd.DataFrame
    pipeline1_report: dict[str, Any] | None
    dataset_name: str
    csv_path: Path


def resolve_cleaned_dataset(
    dataset_id: str,
    *,
    project_root: Path | None = None,
    runtime_root: Path | None = None,
) -> ResolvedDataset:
    """Resolve dataset_id to an already-cleaned CSV and optional report on disk.

    Raises ServiceError(404, "dataset_not_found", ...) if not found.
    """
    clean_id = (dataset_id or "").strip()
    if not clean_id:
        raise ServiceError(
            404,
            "dataset_not_found",
            "Dataset ID must be a non-empty string.",
        )

    root = (project_root or _PROJECT_ROOT).resolve()

    # -----------------------------------------------------------------------
    # 1. Application Run UUID check
    # -----------------------------------------------------------------------
    is_uuid = False
    try:
        UUID(clean_id)
        is_uuid = True
    except (ValueError, TypeError, AttributeError):
        pass

    if is_uuid and runtime_root is not None:
        run_p1 = Path(runtime_root) / clean_id / "pipeline1"
        cleaned_csv = run_p1 / "cleaned_dataset.csv"
        if cleaned_csv.is_file():
            report_file = run_p1 / "cleaning_report.json"
            report_data: dict[str, Any] | None = None
            if report_file.is_file():
                try:
                    report_data = json.loads(report_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            try:
                df = pd.read_csv(cleaned_csv)
            except Exception as exc:
                raise ServiceError(
                    500,
                    "dataset_read_error",
                    f"Failed to read cleaned dataset for run '{clean_id}': {exc}",
                ) from exc
            return ResolvedDataset(
                dataframe=df,
                pipeline1_report=report_data,
                dataset_name=clean_id,
                csv_path=cleaned_csv,
            )

    # -----------------------------------------------------------------------
    # 2. TabArena task output check
    # -----------------------------------------------------------------------
    tabarena_id = clean_id
    for prefix in ("tabarena:", "tabarena/", "task_"):
        if tabarena_id.lower().startswith(prefix):
            tabarena_id = tabarena_id[len(prefix):]

    tab_output_dir = root / "benchmarks" / "tabarena" / "pipeline1_outputs" / tabarena_id
    tab_csv = tab_output_dir / "cleaned_dataset.csv"
    if tab_csv.is_file():
        report_file = tab_output_dir / "cleaning_report.json"
        report_data = None
        if report_file.is_file():
            try:
                report_data = json.loads(report_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        try:
            df = pd.read_csv(tab_csv)
        except Exception as exc:
            raise ServiceError(
                500,
                "dataset_read_error",
                f"Failed to read TabArena cleaned dataset '{clean_id}': {exc}",
            ) from exc
        return ResolvedDataset(
            dataframe=df,
            pipeline1_report=report_data,
            dataset_name=f"tabarena_{tabarena_id}",
            csv_path=tab_csv,
        )

    # -----------------------------------------------------------------------
    # 3. Domain datasets check
    #    Priority: storage/domain_datasets/ (production) -> tests/datasets/valid/ (fallback)
    # -----------------------------------------------------------------------
    lookup_key = clean_id.lower()
    candidate_filename = _DOMAIN_ALIASES.get(lookup_key, clean_id)
    if not candidate_filename.endswith(".csv"):
        candidate_filename = f"{candidate_filename}.csv"

    # Candidate paths in priority order: production dir first, test fixture as fallback
    _domain_candidates = [
        root / "storage" / "domain_datasets" / candidate_filename,
        root / "tests" / "datasets" / "valid" / candidate_filename,
    ]
    # Also try direct match (no alias / no .csv appended)
    if not clean_id.endswith(".csv"):
        _domain_candidates += [
            root / "storage" / "domain_datasets" / clean_id,
            root / "tests" / "datasets" / "valid" / clean_id,
        ]
    else:
        _domain_candidates += [
            root / "storage" / "domain_datasets" / clean_id,
            root / "tests" / "datasets" / "valid" / clean_id,
        ]

    domain_csv: Path | None = None
    for _candidate in _domain_candidates:
        if _candidate.is_file():
            domain_csv = _candidate
            break

    if domain_csv is not None:
        try:
            df = pd.read_csv(domain_csv)
        except Exception as exc:
            raise ServiceError(
                500,
                "dataset_read_error",
                f"Failed to read domain dataset '{clean_id}': {exc}",
            ) from exc
        return ResolvedDataset(
            dataframe=df,
            pipeline1_report=None,
            dataset_name=domain_csv.stem,
            csv_path=domain_csv,
        )

    # -----------------------------------------------------------------------
    # 4. storage/cleaned check
    # -----------------------------------------------------------------------
    storage_cleaned = root / "storage" / "cleaned"
    if storage_cleaned.is_dir():
        for candidate in (storage_cleaned / f"{clean_id}.csv", storage_cleaned / clean_id / "cleaned_dataset.csv"):
            if candidate.is_file():
                try:
                    df = pd.read_csv(candidate)
                except Exception as exc:
                    raise ServiceError(
                        500,
                        "dataset_read_error",
                        f"Failed to read storage dataset '{clean_id}': {exc}",
                    ) from exc
                return ResolvedDataset(
                    dataframe=df,
                    pipeline1_report=None,
                    dataset_name=candidate.stem,
                    csv_path=candidate,
                )

    # -----------------------------------------------------------------------
    # 5. Not found
    # -----------------------------------------------------------------------
    raise ServiceError(
        404,
        "dataset_not_found",
        f"Dataset '{dataset_id}' could not be resolved to a known cleaned dataset on disk.",
    )