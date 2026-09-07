from pprint import pprint

from intelligence.llm.mock_provider import (
    MockLLMProvider,
)

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
    understand_column,
)


FILE_PATH = (
    r"G:\My Drive\LYNX\tests\datasets\valid\pizza_sales.xlsx"
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

dataset_profile = profile_dataset(
    dataframe
)


# ============================================================
# Select order_id
# ============================================================

order_id_profile = next(
    column
    for column
    in dataset_profile[
        "column_profiles"
    ]
    if column[
        "column_name"
    ] == "order_id"
)


# ============================================================
# Evidence
# ============================================================

evidence = build_column_evidence(
    dataset_profile=dataset_profile,
    column_profile=order_id_profile
)


# ============================================================
# Semantic components
# ============================================================

retriever = MockRAGRetriever()

llm = MockLLMProvider()


# ============================================================
# Understand column
# ============================================================

semantic_result = understand_column(
    evidence=evidence,
    retriever=retriever,
    llm_provider=llm
)


print(
    "\nCOLUMN SEMANTIC RESULT"
)

print("=" * 60)

pprint(
    semantic_result.model_dump()
)