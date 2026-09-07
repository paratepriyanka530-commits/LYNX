from pprint import pprint

from intelligence.llm.ollama_provider import OllamaProvider
from intelligence.rag.retriever import RAGRetriever

from pipelines.pipeline1_cleaning.ingestion.dataset_loader import (
    load_dataset,
)

from pipelines.pipeline1_cleaning.profiling.dataset_profiler import (
    profile_dataset,
)

from pipelines.pipeline1_cleaning.semantics.evidence_builder import (
    build_dataset_evidence,
    build_column_evidence,
)

from pipelines.pipeline1_cleaning.semantics.dataset_understanding import (
    understand_dataset,
)

from pipelines.pipeline1_cleaning.semantics.column_understanding import (
    retrieve_column_context,
    understand_column,
)

from pipelines.pipeline1_cleaning.semantics.role_resolver import (
    retrieve_role_context,
)

from pipelines.pipeline1_cleaning.semantics.role_resolver import (
    retrieve_role_context,
    resolve_column_role,
)


# --------------------------------------------------
# 1. Dataset + target column
# --------------------------------------------------

DATASET_PATH = "tests/datasets/valid/dataset_03_healthcare_raw.csv"

TARGET_COLUMN = "patient_id"


# --------------------------------------------------
# 2. Load + profile dataset
# --------------------------------------------------

loaded = load_dataset(DATASET_PATH)

dataframe = loaded["dataframe"]

dataset_profile = profile_dataset(
    dataframe
)


# --------------------------------------------------
# 3. Initialise RAG
# --------------------------------------------------

retriever = RAGRetriever(
    knowledge_directory="knowledge_base"
)

print("\nIndexing knowledge base...")

retriever.index_knowledge()

print("Knowledge base indexed.")


# --------------------------------------------------
# 4. Initialise LLM
# --------------------------------------------------

llm_provider = OllamaProvider(
    model="qwen2.5:3b"
)


# --------------------------------------------------
# 5. Dataset understanding
# --------------------------------------------------

dataset_evidence = build_dataset_evidence(
    dataset_profile
)

dataset_semantics = understand_dataset(
    evidence=dataset_evidence,
    retriever=retriever,
    llm_provider=llm_provider,
    top_k=8,
)

print("\n" + "=" * 70)
print("DATASET SEMANTIC CONTEXT")
print("=" * 70)

pprint(
    dataset_semantics.model_dump()
)


# --------------------------------------------------
# 6. Build evidence for ONE column
# --------------------------------------------------

column_profile = next(
    column
    for column in dataset_profile["column_profiles"]
    if column["column_name"] == TARGET_COLUMN
)

column_evidence_packet = build_column_evidence(
    dataset_profile=dataset_profile,
    column_profile=column_profile,
)

column_evidence = column_evidence_packet["column"]

# --------------------------------------------------
# 7. Inspect retrieved column knowledge
# --------------------------------------------------

rag_context = retrieve_column_context(
    column_evidence=column_evidence,
    dataset_semantics=dataset_semantics,
    retriever=retriever,
    top_k=6,
)

print("\n" + "=" * 70)
print("SEMANTIC CONTEXT")
print("=" * 70)

for index, item in enumerate(
    rag_context["semantic_context"],
    start=1,
):
    print(f"\nResult {index}")
    print(f"Score: {item['score']:.4f}")
    print(f"Category: {item['category']}")
    print(f"Topic: {item['topic']}")
    print(f"Section: {item['section']}")
    print(f"Source: {item['source']}")
    print("Content:")
    print(item["content"][:500])
    print("-" * 70)


print("\n" + "=" * 70)
print("VALUE-PATTERN CONTEXT")
print("=" * 70)

for index, item in enumerate(
    rag_context["pattern_context"],
    start=1,
):
    print(f"\nResult {index}")
    print(f"Score: {item['score']:.4f}")
    print(f"Category: {item['category']}")
    print(f"Topic: {item['topic']}")
    print(f"Section: {item['section']}")
    print(f"Source: {item['source']}")
    print("Content:")
    print(item["content"][:500])
    print("-" * 70)


# --------------------------------------------------
# 8. Column understanding
# --------------------------------------------------
print("\nDATASET SEMANTICS PASSED TO COLUMN UNDERSTANDING")
print("=" * 70)
pprint(dataset_semantics.model_dump())

result = understand_column(
    column_evidence=column_evidence,
    dataset_semantics=dataset_semantics,
    retriever=retriever,
    llm_provider=llm_provider,
    top_k=6,
)

# --------------------------------------------------
# 8.1 Inspect dedicated role resolver knowledge
# --------------------------------------------------

role_context = retrieve_role_context(
    column_evidence=column_evidence,
    column_semantics=result,
    dataset_semantics=dataset_semantics,
    retriever=retriever,
    top_k=5,
)

print("\n" + "=" * 70)
print("ROLE RESOLVER CONTEXT")
print("=" * 70)

for index, item in enumerate(role_context, start=1):
    print(f"\nResult {index}")
    print(f"Score: {item['score']:.4f}")
    print(f"Category: {item['category']}")
    print(f"Topic: {item['topic']}")
    print(f"Section: {item['section']}")
    print(f"Source: {item['source']}")
    print("Content:")
    print(item["content"][:500])
    print("-" * 70)

role_result = resolve_column_role(
    column_evidence=column_evidence,
    column_semantics=result,
    dataset_semantics=dataset_semantics,
    retriever=retriever,
    llm=llm_provider,
    top_k=5,
)

print("\n" + "=" * 70)
print("ROLE RESOLVER RESULT")
print("=" * 70)

pprint(role_result.model_dump())

# --------------------------------------------------
# 9. Final result
# --------------------------------------------------

print("\n" + "=" * 70)
print("COLUMN SEMANTIC RESULT")
print("=" * 70)

pprint(
    result.model_dump()
)