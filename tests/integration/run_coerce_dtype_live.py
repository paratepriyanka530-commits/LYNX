"""Single-dataset live audit using the frozen default semantic/decision stages."""

import hashlib
import json
import time
from pathlib import Path

from pipelines.pipeline1_cleaning.pipeline import (
    _run_default_decision_engine, _run_default_semantic_analyzer, run_pipeline1,
)


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / "tests/datasets/valid/dataset_13_generic_business_data_raw.csv"
    destination = root / "tests/integration/pipeline1_coerce_dtype_business_output"
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
    instructions = [item for item in report.instructions if item.action_id == "coerce_dtype"]
    executions = [item for item in report.executions if item.action_id == "coerce_dtype"]
    routing = report.decision_summary["execution_routing"]
    summary = {
        "pipeline_status": report.status,
        "mixed_numeric_candidates": sum(item["issue_type"] == "mixed_numeric_representation_candidate" for item in report.issues),
        "coerce_dtype_proposals": sum(item.proposed_action == "coerce_dtype" for item in report.decisions),
        "compiled_instructions": [item.model_dump(mode="json") for item in instructions],
        "skipped_proposals": [item for item in routing["skipped_actions"] if item["proposed_action"] == "coerce_dtype"],
        "numeric_assessments": routing["numeric_coercion_assessments"],
        "successful_coercions": sum(item.status == "success" for item in executions),
        "cells_converted": sum(item.changed_cell_count for item in executions),
        "total_cells_changed": report.validation.changed_cell_count,
        "before_issue_counts": report.validation.before_issue_counts,
        "after_issue_counts": report.validation.after_issue_counts,
        "before_summary": report.validation.before_summary,
        "after_summary": report.validation.after_summary,
        "new_null_count": report.validation.newly_introduced_null_count,
        "null_positions_equal": report.validation.null_positions_equal,
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
