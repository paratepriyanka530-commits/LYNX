from pprint import pprint
import re

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
    build_column_semantic_query,
    build_value_pattern_query,
    retrieve_column_context,
    understand_column,
)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

dataset_path = "tests/datasets/valid/pizza_sales.xlsx"

loaded = load_dataset(dataset_path)
dataframe = loaded["dataframe"]


# --------------------------------------------------
# 2. Profile dataset
# --------------------------------------------------

dataset_profile = profile_dataset(dataframe)
print("\nDATASET PROFILE KEYS")
print(dataset_profile.keys())

# --------------------------------------------------
# 3. Build dataset evidence
# --------------------------------------------------

dataset_evidence = build_dataset_evidence(
    dataset_profile
)


# --------------------------------------------------
# 4. Initialize RAG
# --------------------------------------------------

retriever = RAGRetriever(
    knowledge_directory="knowledge_base/semantic"
)

retriever.index_knowledge()

print(
    f"\nIndexed semantic chunks: "
    f"{retriever.vector_store.count()}"
)


# --------------------------------------------------
# 5. Initialize Ollama
# --------------------------------------------------

llm_provider = OllamaProvider(
    model="qwen2.5:3b"
)


# --------------------------------------------------
# 6. Understand dataset first
# --------------------------------------------------

dataset_semantics = understand_dataset(
    evidence=dataset_evidence,
    retriever=retriever,
    llm_provider=llm_provider,
    top_k=8,
)

print("\nDATASET SEMANTICS")
print("=" * 80)

pprint(
    dataset_semantics.model_dump()
)

TARGET_COLUMN = "quantity"
# --------------------------------------------------
# 7. Find order_id profile
# --------------------------------------------------

target_column_profile = None

for column_profile in dataset_evidence["columns"]:
    if column_profile["column_name"] == TARGET_COLUMN:
        target_column_profile = column_profile
        break

if target_column_profile is None:
    raise ValueError(
        f"{TARGET_COLUMN} column evidence was not found."
    )

column_evidence = target_column_profile


# --------------------------------------------------
# 8. Build controlled column evidence
# --------------------------------------------------

column_evidence = target_column_profile

print("\nORDER_DATE EVIDENCE")
print("=" * 80)

pprint(column_evidence)

# --------------------------------------------------
# 9. Inspect RAG query
# --------------------------------------------------

query = build_column_semantic_query(
    column_evidence=column_evidence,
    dataset_semantics=dataset_semantics,
)

print("\nCOLUMN RAG QUERY")
print("=" * 80)

print(query)


# --------------------------------------------------
# 10. Retrieve semantic knowledge
# --------------------------------------------------

rag_results = retrieve_column_context(
    column_evidence=column_evidence,
    dataset_semantics=dataset_semantics,
    retriever=retriever,
    top_k=6,
)

print("\nSEMANTIC CONTEXT RAG RESULTS")
print("=" * 80)

for result in rag_results["semantic_context"]:
    print(
        f"{result['chunk_id']} | "
        f"{result['topic']:<20} | "
        f"{result['section']:<25} | "
        f"{result['score']:.4f}"
    )

    print(result["content"][:300])
    print("-" * 80)


print("\nPATTERN CONTEXT RAG RESULTS")
print("=" * 80)

for result in rag_results["pattern_context"]:
    print(
        f"{result['chunk_id']} | "
        f"{result['topic']:<20} | "
        f"{result['section']:<25} | "
        f"{result['score']:.4f}"
    )

    print(result["content"][:300])
    print("-" * 80)


# --------------------------------------------------
# 11. Run column understanding
# --------------------------------------------------

result = understand_column(
    column_evidence=column_evidence,
    dataset_semantics=dataset_semantics,
    retriever=retriever,
    llm_provider=llm_provider,
    top_k=6,
)


# --------------------------------------------------
# 12. Display final result
# --------------------------------------------------

print("\nORDER_DATE SEMANTIC RESULT")
print("=" * 80)

pprint(
    result.model_dump()
)
