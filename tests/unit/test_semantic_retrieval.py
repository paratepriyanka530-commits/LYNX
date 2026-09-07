from pprint import pprint

from intelligence.rag.mock_retriever import (
    MockRAGRetriever,
)

from pipelines.pipeline1_cleaning.ingestion.dataset_loader import (
    load_dataset,
)

from pipelines.pipeline1_cleaning.profiling.dataset_profiler import (
    profile_dataset,
)

from pipelines.pipeline1_cleaning.semantics.evidence_builder import (
    build_column_evidence,
)

from pipelines.pipeline1_cleaning.semantics.column_understanding import (
    retrieve_semantic_context,
)


FILE_PATH = (
    r"G:\My Drive\LYNX\tests\datasets\valid\pizza_sales.xlsx"
)


# ------------------------------------------------------------
# Load + profile
# ------------------------------------------------------------

load_result = load_dataset(
    FILE_PATH
)

dataframe = load_result["dataframe"]

profile = profile_dataset(
    dataframe
)


# ------------------------------------------------------------
# Find order_id
# ------------------------------------------------------------

column_profile = next(
    column
    for column in profile["column_profiles"]
    if column["column_name"] == "order_id"
)


# ------------------------------------------------------------
# Build evidence
# ------------------------------------------------------------

evidence = build_column_evidence(
    dataset_profile=profile,
    column_profile=column_profile
)


# ------------------------------------------------------------
# Retrieve knowledge
# ------------------------------------------------------------

retriever = MockRAGRetriever()

semantic_context = retrieve_semantic_context(
    evidence=evidence,
    retriever=retriever
)


print("\nSEMANTIC RETRIEVAL")
print("=" * 60)

pprint(
    semantic_context
)