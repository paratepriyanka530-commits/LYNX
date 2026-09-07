from pprint import pprint

from pipelines.pipeline1_cleaning.ingestion.dataset_loader import (
    load_dataset,
)

from pipelines.pipeline1_cleaning.profiling.dataset_profiler import (
    profile_dataset,
)


FILE_PATH = (
    r"G:\My Drive\LYNX\tests\datasets\valid\pizza_sales.xlsx"
)


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

load_result = load_dataset(
    FILE_PATH
)

dataframe = load_result[
    "dataframe"
]


# ------------------------------------------------------------
# Profile
# ------------------------------------------------------------

profile = profile_dataset(
    dataframe
)


# ------------------------------------------------------------
# Display dataset-level evidence
# ------------------------------------------------------------

print("\nDATASET PROFILE")
print("=" * 60)

print(
    "Shape:",
    profile["shape"]
)

print(
    "Dtypes:",
    profile["dtype_distribution"]
)

print(
    "Missingness:",
    profile["missingness"]
)

print(
    "Duplicates:",
    profile["duplicates"]
)

print(
    "\nColumns:"
)

pprint(
    profile["column_names"]
)

print(
    "\nNumber of column profiles:",
    len(profile["column_profiles"])
)