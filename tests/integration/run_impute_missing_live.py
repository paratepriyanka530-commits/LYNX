"""Single business-dataset imputation audit with frozen semantic/decision stages."""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from pipelines.pipeline1_cleaning.pipeline import (
    _run_default_decision_engine, _run_default_semantic_analyzer, run_pipeline1,
)


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / "tests/datasets/valid/dataset_13_generic_business_data_raw.csv"
    destination = root / "tests/integration/pipeline1_impute_missing_business_output"
    raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    def semantic_stage(*args):
        print("Starting unchanged Architecture B semantics", flush=True)
        started = time.perf_counter()
        result = _run_default_semantic_analyzer(*args)
        print(f"Semantics completed in {time.perf_counter() - started:.2f}s", flush=True)
        return result

    def decision_stage(*args):
        print("Starting frozen Decision Engine", flush=True)
        started = time.perf_counter()
        result = _run_default_decision_engine(*args)
        print(f"Decisions completed in {time.perf_counter() - started:.2f}s", flush=True)
        return result

    report = run_pipeline1(source, destination, semantic_analyzer=semantic_stage, decision_engine=decision_stage)
    proposals = [item for item in report.decisions if item.proposed_action == "impute_missing"]
    instructions = [item for item in report.instructions if item.action_id == "impute_missing"]
    executions = [item for item in report.executions if item.action_id == "impute_missing"]
    routing = report.decision_summary["execution_routing"]
    by_strategy = {strategy: 0 for strategy in ("median", "mode", "constant")}
    for item in executions:
        if item.status == "success":
            by_strategy[item.audit_metadata["strategy"]] += item.audit_metadata["affected_cell_count"]
    summary = {
        "pipeline_status": report.status, "validation_passed": report.validation.passed,
        "missing_value_issue_count": sum(item["issue_type"] == "missing_values" for item in report.issues),
        "missing_columns": routing["missing_value_assessments"],
        "impute_missing_proposals": len(proposals),
        "proposed_strategy_distribution": dict(Counter(item.parameters.get("strategy", "unspecified") for item in proposals)),
        "compiled_instructions": [item.model_dump(mode="json") for item in instructions],
        "skipped_proposals": [item for item in routing["skipped_actions"] if item["proposed_action"] == "impute_missing"],
        "successful_imputations": sum(item.status == "success" for item in executions),
        "executions": [item.model_dump(mode="json") for item in executions],
        "imputed_cells_by_strategy": by_strategy,
        "cells_imputed": report.validation.imputed_cell_count,
        "total_cells_changed": report.validation.changed_cell_count,
        "before_summary": report.validation.before_summary,
        "after_summary": report.validation.after_summary,
        "new_null_count": report.validation.newly_introduced_null_count,
        "expected_missingness_changes_only": report.validation.expected_missingness_changes_only,
        "unrelated_columns_equal": report.validation.unrelated_columns_equal,
        "raw_sha256_before": raw_hash,
        "raw_sha256_after": hashlib.sha256(source.read_bytes()).hexdigest(),
        "artifacts": report.output_artifacts,
    }
    (destination / "live_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    assert summary["raw_sha256_before"] == summary["raw_sha256_after"]
    assert report.validation.newly_introduced_null_count == 0


if __name__ == "__main__":
    main()
