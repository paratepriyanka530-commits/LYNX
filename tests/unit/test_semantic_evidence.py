from pprint import pprint

from pipelines.pipeline1_cleaning.ingestion.dataset_loader import (
    load_dataset,
)

from pipelines.pipeline1_cleaning.profiling.dataset_profiler import (
    profile_dataset,
)

from pipelines.pipeline1_cleaning.semantics.evidence_builder import (
    build_column_evidence,
    build_dataset_evidence,
)


FILE_PATH = (
    r"G:\My Drive\LYNX\tests\datasets\valid\pizza_sales.xlsx"
)


load_result = load_dataset(
    FILE_PATH
)

dataframe = load_result[
    "dataframe"
]

profile = profile_dataset(
    dataframe
)


print("\nDATASET EVIDENCE")
print("=" * 60)

dataset_evidence = build_dataset_evidence(
    profile
)

pprint(
    dataset_evidence
)


print("\nCOLUMN EVIDENCE: order_id")
print("=" * 60)

order_id_profile = next(
    column
    for column in profile["column_profiles"]
    if column["column_name"] == "order_id"
)

column_evidence = build_column_evidence(
    dataset_profile=profile,
    column_profile=order_id_profile
)

pprint(
    column_evidence
)