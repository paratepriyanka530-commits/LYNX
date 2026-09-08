import openml

FAILED_TASKS = [
    363705,
    363706,
    363707,
    363708,
    363711,
    363712,
]

for task_id in FAILED_TASKS:
    try:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        print(
            f"{task_id} -> {dataset.name}"
        )

    except Exception as e:
        print(
            f"{task_id} -> ERROR: {e}"
        )