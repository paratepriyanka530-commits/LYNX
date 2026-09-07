from pathlib import Path
from collections import Counter
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "tests"
    / "pipeline1_multidomain"
    / "results"
)


DATASETS = [
    "dataset_01_retail_transactions",
    "dataset_02_hr_employees",
    "dataset_03_healthcare",
    "dataset_04_banking_finance",
    "dataset_05_education",
    "dataset_06_cybersecurity",
    "dataset_07_iot_sensor",
    "dataset_08_manufacturing",
    "dataset_09_survey_data",
    "dataset_10_ecommerce",
    "dataset_11_scientific_research",
    "dataset_12_log_event_data",
    "dataset_13_generic_business_data",
]


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )


def main():

    dataset_records = []
    issue_records = []
    decision_records = []
    execution_records = []
    validation_records = []

    issues_by_type = Counter()
    decisions_by_type = Counter()
    proposed_actions = Counter()
    executions_by_action = Counter()

    total_rows = 0
    total_columns = 0
    total_issues = 0
    total_decisions = 0
    total_executions = 0
    total_cells_changed = 0
    total_runtime = 0.0

    successful = 0
    failed = 0
    validation_passed = 0
    validation_failed = 0
    raw_preserved = 0

    for dataset_id in DATASETS:

        dataset_dir = RESULTS_DIR / dataset_id

        report_path = dataset_dir / "cleaning_report.json"
        checkpoint_path = dataset_dir / "multidomain_checkpoint.json"

        if not report_path.exists():
            print(f"MISSING REPORT: {dataset_id}")
            failed += 1
            continue

        with report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)

        checkpoint = {}

        if checkpoint_path.exists():
            with checkpoint_path.open("r", encoding="utf-8") as file:
                checkpoint = json.load(file)

        original = report.get("original_dataset", {})
        validation = report.get("validation", {})

        issues = report.get("issues", [])
        decisions = report.get("decisions", [])
        executions = report.get("executions", [])

        rows = original.get("rows", 0)
        columns = original.get("columns", 0)

        runtime = checkpoint.get("runtime_seconds", 0.0)

        changed_cells = validation.get(
            "changed_cell_count",
            sum(
                execution.get("changed_cell_count", 0) or 0
                for execution in executions
            ),
        )

        # ----------------------------------------------
        # Dataset-level record
        # ----------------------------------------------

        dataset_record = {
            "dataset_id": dataset_id,
            "file_name": original.get("file_name"),
            "pipeline_version": report.get("pipeline_version"),
            "run_id": report.get("run_id"),
            "status": report.get("status"),
            "rows": rows,
            "columns": columns,
            "runtime_seconds": runtime,
            "issues_detected": len(issues),
            "decisions": len(decisions),
            "executions": len(executions),
            "cells_changed": changed_cells,
            "validation_passed": validation.get("passed"),
            "raw_dataset_overwritten": original.get(
                "raw_dataset_overwritten"
            ),
            "warning_count": len(report.get("warnings", [])),
        }

        dataset_records.append(dataset_record)

        # ----------------------------------------------
        # Issues
        # ----------------------------------------------

        for issue in issues:
            issue_type = issue.get("issue_type", "unknown")
            issues_by_type[issue_type] += 1

            issue_records.append({
                "dataset_id": dataset_id,
                "run_id": report.get("run_id"),
                **issue,
            })

        # ----------------------------------------------
        # Decisions
        # ----------------------------------------------

        for decision in decisions:
            decision_type = decision.get(
                "decision_type",
                "unknown",
            )

            action = decision.get(
                "proposed_action",
                "unknown",
            )

            decisions_by_type[decision_type] += 1
            proposed_actions[action] += 1

            decision_records.append({
                "dataset_id": dataset_id,
                "run_id": report.get("run_id"),
                **decision,
            })

        # ----------------------------------------------
        # Executions
        # ----------------------------------------------

        for execution in executions:
            action = execution.get(
                "action_id",
                "unknown",
            )

            executions_by_action[action] += 1

            execution_records.append({
                "dataset_id": dataset_id,
                "run_id": report.get("run_id"),
                **execution,
            })

        # ----------------------------------------------
        # Validation
        # ----------------------------------------------

        validation_records.append({
            "dataset_id": dataset_id,
            "run_id": report.get("run_id"),
            **validation,
        })

        # ----------------------------------------------
        # Global totals
        # ----------------------------------------------

        total_rows += rows
        total_columns += columns
        total_issues += len(issues)
        total_decisions += len(decisions)
        total_executions += len(executions)
        total_cells_changed += changed_cells
        total_runtime += runtime

        if report.get("status") == "success":
            successful += 1
        else:
            failed += 1

        if validation.get("passed"):
            validation_passed += 1
        else:
            validation_failed += 1

        if original.get("raw_dataset_overwritten") is False:
            raw_preserved += 1

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    summary = {
        "experiment": "LYNX Pipeline 1 Multi-Domain Evaluation",
        "pipeline_version": (
            dataset_records[0]["pipeline_version"]
            if dataset_records
            else None
        ),

        "datasets_attempted": len(DATASETS),
        "datasets_successful": successful,
        "datasets_failed": failed,

        "total_rows_processed": total_rows,
        "total_columns_processed": total_columns,

        "total_issues_detected": total_issues,
        "total_decisions": total_decisions,
        "total_actions_executed": total_executions,
        "total_cells_changed": total_cells_changed,

        "issues_by_type": dict(issues_by_type),
        "decisions_by_type": dict(decisions_by_type),
        "proposed_actions": dict(proposed_actions),
        "executions_by_action": dict(executions_by_action),

        "validation_passed": validation_passed,
        "validation_failed": validation_failed,

        "raw_files_preserved": raw_preserved,

        "total_runtime_seconds": round(total_runtime, 3),
        "total_runtime_minutes": round(
            total_runtime / 60,
            2,
        ),
    }

    # --------------------------------------------------
    # Save consolidated files
    # --------------------------------------------------

    with (
        RESULTS_DIR / "summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    write_jsonl(
        RESULTS_DIR / "datasets.jsonl",
        dataset_records,
    )

    write_jsonl(
        RESULTS_DIR / "issues.jsonl",
        issue_records,
    )

    write_jsonl(
        RESULTS_DIR / "decisions.jsonl",
        decision_records,
    )

    write_jsonl(
        RESULTS_DIR / "executions.jsonl",
        execution_records,
    )

    write_jsonl(
        RESULTS_DIR / "validation.jsonl",
        validation_records,
    )

    print("=" * 70)
    print("CONSOLIDATION COMPLETE")
    print("=" * 70)

    print(f"Datasets:       {successful}/{len(DATASETS)} successful")
    print(f"Rows:           {total_rows:,}")
    print(f"Columns:        {total_columns:,}")
    print(f"Issues:         {total_issues:,}")
    print(f"Decisions:      {total_decisions:,}")
    print(f"Executions:     {total_executions:,}")
    print(f"Cells changed:  {total_cells_changed:,}")
    print(
        f"Validation:     "
        f"{validation_passed}/{len(DATASETS)} passed"
    )
    print(f"Raw preserved:  {raw_preserved}/{len(DATASETS)}")
    print(
        f"Runtime:        "
        f"{total_runtime / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()