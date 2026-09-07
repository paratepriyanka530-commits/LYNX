"""One-dataset live parse_dates audit; uses the unchanged default LLM stages."""

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
    destination = root / "tests/integration/pipeline1_parse_dates_business_output"
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
    proposals = [item for item in report.decisions if item.proposed_action == "parse_dates"]
    executions = [item for item in report.executions if item.action_id == "parse_dates"]
    routing = report.decision_summary["execution_routing"]
    summary = {
        "pipeline_status": report.status,
        "date_format_candidates": sum(item["issue_type"] == "inconsistent_date_format_candidate" for item in report.issues),
        "parse_proposals": len(proposals),
        "compiled_parse_instructions": sum(item.action_id == "parse_dates" for item in report.instructions),
        "skipped_parse_proposals": [item for item in routing["skipped_actions"] if item["proposed_action"] == "parse_dates"],
        "parse_coverage": routing["parse_date_assessments"],
        "successful_parse_executions": sum(item.status == "success" for item in executions),
        "date_cells_changed": sum(item.changed_cell_count for item in executions),
        "date_columns": sorted({column for item in executions for column in item.target_columns}),
        "total_cells_changed": report.validation.changed_cell_count,
        "before_issue_counts": report.validation.before_issue_counts,
        "after_issue_counts": report.validation.after_issue_counts,
        "before_summary": report.validation.before_summary,
        "after_summary": report.validation.after_summary,
        "new_null_count": report.validation.newly_introduced_null_count,
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
