import json
import subprocess
import sys
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

TARGET_REGISTRY = (
    ROOT
    / "pipelines"
    / "pipeline2_prediction"
    / "inputs"
    / "dataset_target_registry.json"
)

RESULT_FILE = (
    ROOT
    / "pipelines"
    / "pipeline2_prediction"
    / "training_data"
    / "model_performance.json"
)


# --------------------------------------------------
# Extract the 51 dataset IDs from the existing registry
# --------------------------------------------------

def load_dataset_ids():
    with open(TARGET_REGISTRY, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_ids = []

    # Registry stored as list
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                dataset_id = (
                    item.get("dataset_id")
                    or item.get("task_id")
                    or item.get("id")
                )

                if dataset_id is not None:
                    dataset_ids.append(str(dataset_id))

    # Registry stored as dictionary
    elif isinstance(data, dict):

        # Common nested structures
        for key in ["datasets", "targets", "records", "items"]:
            if key in data and isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict):
                        dataset_id = (
                            item.get("dataset_id")
                            or item.get("task_id")
                            or item.get("id")
                        )

                        if dataset_id is not None:
                            dataset_ids.append(str(dataset_id))

                break

        # Registry keyed directly by dataset ID
        if not dataset_ids:
            for key, value in data.items():
                if str(key).isdigit():
                    dataset_ids.append(str(key))

                elif isinstance(value, dict):
                    dataset_id = (
                        value.get("dataset_id")
                        or value.get("task_id")
                        or value.get("id")
                    )

                    if dataset_id is not None:
                        dataset_ids.append(str(dataset_id))

    # Remove duplicates while preserving order
    dataset_ids = list(dict.fromkeys(dataset_ids))

    if not dataset_ids:
        raise RuntimeError(
            "Could not find dataset IDs in dataset_target_registry.json"
        )

    return dataset_ids


# --------------------------------------------------
# Optional clean start
# --------------------------------------------------

def reset_results():
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if RESULT_FILE.exists():
        RESULT_FILE.unlink()

    print("\nStarting clean benchmark.")
    print(f"Results will be saved to:")
    print(RESULT_FILE)


# --------------------------------------------------
# Run benchmark dataset-by-dataset
# --------------------------------------------------

def run():
    dataset_ids = load_dataset_ids()

    print(f"\nFound {len(dataset_ids)} datasets.")

    if len(dataset_ids) != 51:
        print(
            f"WARNING: Expected 51 datasets but found {len(dataset_ids)}."
        )

    total_models = 10
    expected_runs = len(dataset_ids) * total_models

    print(f"Expected model evaluations: {expected_runs}")
    print(f"Results: {RESULT_FILE}\n")

    for dataset_number, dataset_id in enumerate(dataset_ids, start=1):

        start_index = ((dataset_number - 1) * total_models) + 1
        end_index = dataset_number * total_models

        print("\n" + "=" * 70)
        print(
            f"DATASET {dataset_number}/{len(dataset_ids)}"
            f" | task_id={dataset_id}"
        )
        print(
            f"Expected model positions: "
            f"{start_index}-{end_index}/{expected_runs}"
        )
        print("=" * 70)

        command = [
            sys.executable,
            "-m",
            "benchmarks.tabarena.pipeline2_benchmark_runner",
            "--dataset",
            str(dataset_id),
        ]

        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
            )

            if result.returncode != 0:
                print(
                    f"\nDataset {dataset_id} benchmark process "
                    f"returned exit code {result.returncode}."
                )

                print(
                    "Continuing to the next dataset. "
                    "Any results already saved by the runner are preserved."
                )

        except KeyboardInterrupt:
            print("\n\nBenchmark stopped by user.")
            print("Already written model results are preserved.")
            sys.exit(0)

        except Exception as exc:
            print(
                f"\nUnexpected error while starting "
                f"dataset {dataset_id}: {exc}"
            )
            print("Continuing to the next dataset.")

    print("\n" + "=" * 70)
    print("51 × 10 BENCHMARK FINISHED")
    print("=" * 70)
    print(f"Results saved in:")
    print(RESULT_FILE)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    # --------------------------------------------------
    # Set this True only when starting the FINAL run
    # completely from zero.
    # --------------------------------------------------

    START_FROM_ZERO = True

    if START_FROM_ZERO:
        reset_results()

    run()