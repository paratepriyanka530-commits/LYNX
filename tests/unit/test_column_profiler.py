from pprint import pprint

from pipelines.pipeline1_cleaning.ingestion.dataset_loader import (
    load_dataset
)

from pipelines.pipeline1_cleaning.profiling.column_profiler import (
    profile_column
)


FILE_PATH = (
    r"G:\My Drive\LYNX\tests\datasets\valid\pizza_sales.xlsx"
)


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

result = load_dataset(
    FILE_PATH
)

dataframe = result["dataframe"]


# ------------------------------------------------------------
# Profile each column
# ------------------------------------------------------------

for column_name in dataframe.columns:

    print("\n" + "=" * 60)
    print(f"COLUMN: {column_name}")
    print("=" * 60)

    profile = profile_column(
        dataframe[column_name]
    )

    pprint(profile)