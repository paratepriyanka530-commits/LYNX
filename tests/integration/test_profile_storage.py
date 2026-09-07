from database.connection import SessionLocal

from database.repositories.dataset_repository import (
    DatasetRepository,
)

from database.repositories.profile_repository import (
    ProfileRepository,
)

from pipelines.pipeline1_cleaning.ingestion.file_validator import (
    validate_file,
)

from pipelines.pipeline1_cleaning.ingestion.dataset_loader import (
    load_dataset,
)

from pipelines.pipeline1_cleaning.profiling.dataset_profiler import (
    profile_dataset,
)


FILE_PATH = (
    r"G:\My Drive\LYNX\tests\datasets\valid\pizza_sales.xlsx"
)


# ============================================================
# Validate
# ============================================================

validation = validate_file(
    FILE_PATH
)

if not validation["valid"]:
    raise RuntimeError(
        validation["reason"]
    )


# ============================================================
# Load
# ============================================================

load_result = load_dataset(
    FILE_PATH
)

dataframe = load_result[
    "dataframe"
]


# ============================================================
# Profile
# ============================================================

profile = profile_dataset(
    dataframe
)


# ============================================================
# Database
# ============================================================

with SessionLocal() as session:

    dataset_repository = DatasetRepository(
        session
    )

    profile_repository = ProfileRepository(
        session
    )

    # --------------------------------------------------------
    # Register this test dataset
    # --------------------------------------------------------

    dataset = dataset_repository.register_dataset(
        original_filename=validation["file_name"],

        stored_path=validation["absolute_path"],

        file_format=validation["file_format"],

        file_size_bytes=validation["file_size_bytes"],

        row_count=load_result["rows"],

        column_count=load_result["columns"],

        status="registered"
    )

    print(
        "\nDataset ID:",
        dataset.dataset_id
    )

    # --------------------------------------------------------
    # Save profiling evidence
    # --------------------------------------------------------

    saved_profile = profile_repository.save_profile(
        dataset_id=dataset.dataset_id,
        profile=profile
    )

    dataset_repository.update_status(
        dataset_id=dataset.dataset_id,
        status="profiled"
    )

    print(
        "Dataset profile saved successfully."
    )

    print(
        "Profile ID:",
        saved_profile.profile_id
    )

    print(
        "Rows:",
        saved_profile.row_count
    )

    print(
        "Columns:",
        saved_profile.column_count
    )