"""Export Pipeline 1 dataframe and audit report artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pipelines.pipeline1_cleaning.decisions.decision_schema import Pipeline1Report


def export_pipeline_outputs(
    dataframe: pd.DataFrame,
    report: Pipeline1Report,
    output_directory: str | Path,
) -> Dict[str, Any]:
    output_dir = Path(output_directory).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "cleaned_dataset.csv"
    report_path = output_dir / "cleaning_report.json"

    dataframe.to_csv(dataset_path, index=False)
    privacy_actions = report.decision_summary.get("privacy_actions")
    if isinstance(privacy_actions, dict):
        privacy_actions["cleaned_dataset_sha256"] = hashlib.sha256(
            dataset_path.read_bytes()
        ).hexdigest()
    report.output_artifacts = {
        "cleaned_dataset": str(dataset_path),
        "cleaning_report": str(report_path),
        "cleaned_dataset_format": "csv",
    }
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report.output_artifacts
