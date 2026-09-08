"""
Download TabArena v0.1 datasets from OpenML.

Creates:
benchmarks/tabarena/datasets/
    dataset_name.csv

Author: LYNX
"""

from pathlib import Path

import openml
import pandas as pd


# TabArena v0.1 OpenML Study
TABARENA_SUITE_ID = 457


def main():

    output_dir = Path(
        "benchmarks/tabarena/datasets"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Loading OpenML Study {TABARENA_SUITE_ID}..."
    )

    suite = openml.study.get_suite(
        TABARENA_SUITE_ID
    )

    task_ids = suite.tasks

    print(
        f"Found {len(task_ids)} tasks."
    )

    metadata_rows = []

    for idx, task_id in enumerate(
        task_ids,
        start=1,
    ):

        try:

            print(
                f"\n[{idx}/{len(task_ids)}] "
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
                f"Failed task {task_id}: "
                f"{error}"
            )

    metadata = pd.DataFrame(
        metadata_rows
    )

    metadata.to_csv(
        output_dir / "metadata.csv",
        index=False,
    )

    print(
        "\nDownload completed."
    )

    print(
        f"Datasets saved in: {output_dir}"
    )


if __name__ == "__main__":
    main()