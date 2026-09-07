from pathlib import Path
import sys
import json
import time
import traceback

# --------------------------------------------------
# Make project root importable
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from pipelines.pipeline1_cleaning.pipeline import run_pipeline1

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "tests" / "datasets" / "valid"

RESULTS_DIR = (
    PROJECT_ROOT
    / "tests"
    / "pipeline1_multidomain"
    / "results"
)


# --------------------------------------------------
# Controlled 13-domain dataset suite
# --------------------------------------------------

DATASETS = [
    "dataset_01_retail_transactions_raw.csv",
    "dataset_02_hr_employees_raw.csv",
    "dataset_03_healthcare_raw.csv",
    "dataset_04_banking_finance_raw.csv",
    "dataset_05_education_raw.csv",
    "dataset_06_cybersecurity_raw.csv",
    "dataset_07_iot_sensor_raw.csv",
    "dataset_08_manufacturing_raw.csv",
    "dataset_09_survey_data_raw.csv",
    "dataset_10_ecommerce_raw.csv",
    "dataset_11_scientific_research_raw.csv",
    "dataset_12_log_event_data_raw.csv",
    "dataset_13_generic_business_data_raw.csv",
]


def main():
    print("=" * 70)
    print("LYNX PIPELINE 1 — 13-DOMAIN END-TO-END TEST")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_start = time.perf_counter()

    completed = 0
    failed = 0
    skipped = 0

    for index, dataset_name in enumerate(DATASETS, start=1):

        dataset_path = DATASET_DIR / dataset_name
        dataset_id = dataset_name.replace("_raw.csv", "")
        output_dir = RESULTS_DIR / dataset_id
        checkpoint_path = output_dir / "multidomain_checkpoint.json"

        print("\n" + "=" * 70)
        print(f"[{index}/13] {dataset_name}")
        print("=" * 70)

        # ----------------------------------------------
        # Resume support
        # ----------------------------------------------

        if checkpoint_path.exists():
            try:
                with checkpoint_path.open("r", encoding="utf-8") as file:
                    existing = json.load(file)

                if existing.get("status") == "success":
                    print("Already completed successfully — SKIPPING")
                    skipped += 1
                    continue

            except (json.JSONDecodeError, OSError):
                print("Existing checkpoint unreadable — rerunning dataset.")

        # ----------------------------------------------
        # Dataset existence
        # ----------------------------------------------

        if not dataset_path.exists():
            print("Dataset missing — FAILED")
            failed += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        print("Running Pipeline 1...")

        start_time = time.perf_counter()

        try:
            report = run_pipeline1(
                input_path=dataset_path,
                output_directory=output_dir,
            )

            runtime_seconds = time.perf_counter() - start_time

            executions = [
                execution.model_dump(mode="json")
                for execution in report.executions
            ]

            successful_executions = [
                execution
                for execution in executions
                if execution.get("status") == "success"
            ]

            cells_changed = sum(
                int(execution.get("changed_cell_count", 0) or 0)
                for execution in successful_executions
            )

            execution_counts = {}

            for execution in successful_executions:
                action = execution.get("action_id", "unknown")
                execution_counts[action] = (
                    execution_counts.get(action, 0) + 1
                )

            checkpoint = {
                "dataset": dataset_name,
                "dataset_id": dataset_id,
                "status": report.status,
                "run_id": report.run_id,

                "runtime_seconds": round(runtime_seconds, 3),

                "rows": report.before_profile.get("row_count"),
                "columns": report.before_profile.get("column_count"),

                "issues": len(report.issues),
                "decisions": len(report.decisions),

                "executions": len(report.executions),
                "successful_executions": len(successful_executions),
                "execution_counts": execution_counts,

                "cells_changed": cells_changed,

                "validation_passed": report.validation.passed,

                "raw_dataset_overwritten": report.original_dataset.get(
                    "raw_dataset_overwritten"
                ),

                "warnings_count": len(report.warnings),
                "warnings": report.warnings,
            }

            with checkpoint_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    checkpoint,
                    file,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )

            if report.status == "success":
                completed += 1
            else:
                failed += 1

            print(f"Status:       {report.status}")
            print(f"Runtime:      {runtime_seconds:.2f}s")
            print(f"Issues:       {len(report.issues)}")
            print(f"Decisions:    {len(report.decisions)}")
            print(f"Executions:   {len(report.executions)}")
            print(f"Cells changed:{cells_changed}")
            print(
                "Validation:   "
                + ("PASSED" if report.validation.passed else "FAILED")
            )

        except Exception as exc:

            runtime_seconds = time.perf_counter() - start_time
            failed += 1

            failure_checkpoint = {
                "dataset": dataset_name,
                "dataset_id": dataset_id,
                "status": "failed",
                "runtime_seconds": round(runtime_seconds, 3),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

            with checkpoint_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    failure_checkpoint,
                    file,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )

            print(f"FAILED: {exc}")

    total_runtime = time.perf_counter() - run_start

    print("\n" + "=" * 70)
    print("MULTI-DOMAIN RUN FINISHED")
    print("=" * 70)

    print(f"Completed this run: {completed}")
    print(f"Skipped/resumed:    {skipped}")
    print(f"Failed:             {failed}")
    print(f"Session runtime:    {total_runtime:.2f} seconds")


if __name__ == "__main__":
    main()