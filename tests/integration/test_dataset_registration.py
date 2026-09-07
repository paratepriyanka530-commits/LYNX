from pathlib import Path

from database.connection import SessionLocal
from database.repositories.dataset_repository import (
    DatasetRepository
)

from pipelines.pipeline1_cleaning.ingestion.file_validator import (
    validate_file
)

from pipelines.pipeline1_cleaning.ingestion.dataset_loader import (
    load_dataset
)


file_path = Path(
    r"G:\My Drive\LYNX\tests\datasets\valid\pizza_sales.xlsx"
)


# ------------------------------------------------------------
# 1. Validate file
# ------------------------------------------------------------

validation = validate_file(
    str(file_path)
)

if not validation["valid"]:
    raise RuntimeError(
        validation["reason"]
    )


# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

load_result = load_dataset(
    str(file_path)
)


# ------------------------------------------------------------
# 3. Register dataset
# ------------------------------------------------------------

with SessionLocal() as session:

    repository = DatasetRepository(
        session
    )

    dataset = repository.register_dataset(
        original_filename=validation["file_name"],
        stored_path=validation["absolute_path"],
        file_format=validation["file_format"],
        file_size_bytes=validation["file_size_bytes"],
        row_count=load_result["rows"],
        column_count=load_result["columns"],
        status="loaded"
    )

    print("\nDataset registered successfully.")

    print(
        "Dataset ID:",
        dataset.dataset_id
    )

    print(
        "File:",
        dataset.original_filename
    )

    print(
        "Rows:",
        dataset.row_count
    )

    print(
        "Columns:",
        dataset.column_count
    )

    print(
        "Status:",
        dataset.status
    )