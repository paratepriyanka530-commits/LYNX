from pathlib import Path

import openml
import pandas as pd


FAILED_TASKS = [
    363705,
    363706,
    363707,
    363708,
    363711,
    363712,
]


def main():

    output_dir = Path(
        "benchmarks/tabarena/datasets"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_rows = []

    for idx, task_id in enumerate(
        FAILED_TASKS,
        start=1,
    ):

        try:

            print(
                f"[{idx}/{len(FAILED_TASKS)}] "
                f"Downloading task {task_id}"
            )

            task = openml.tasks.get_task(
                task_id
            )

            dataset = task.get_dataset()

            X, y, _, _ = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format="dataframe",
            )

            data = X.copy()

            target_name = (
                dataset.default_target_attribute
            )

            data[target_name] = y

            safe_name = (
                dataset.name
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )

            csv_path = (
                output_dir
                / f"{safe_name}.csv"
            )

            if csv_path.exists():

                print(
                    f"Already exists: {csv_path.name}"
                )

                continue

            data.to_csv(
                csv_path,
                index=False
            )

            metadata_rows.append(
                {
                    "task_id": task_id,
                    "dataset_name": dataset.name,
                    "target_column": target_name,
                    "rows": len(data),
                    "features": len(X.columns),
                    "csv_file": csv_path.name,
                }
            )

            print(
                f"Saved: {csv_path.name}"
            )

        except Exception as error:

            print(
                f"Failed task {task_id}: {error}"
            )

    if metadata_rows:

        metadata = pd.DataFrame(
            metadata_rows
        )

        metadata.to_csv(
            output_dir / "missing_metadata.csv",
            index=False,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()