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
)

from pipelines.pipeline1_cleaning.semantics.dataset_understanding import (
    build_dataset_semantic_query,
    retrieve_dataset_context,
    understand_dataset,
)


# --------------------------------------------------
# 1. Dataset path
# --------------------------------------------------

DATASET_PATH = "tests/datasets/valid/dataset_03_healthcare_raw.csv"


# --------------------------------------------------
# 2. Load + profile dataset
# --------------------------------------------------

loaded = load_dataset(DATASET_PATH)

dataframe = loaded["dataframe"]

dataset_profile = profile_dataset(
    dataframe
)

dataset_evidence = build_dataset_evidence(
    dataset_profile
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
# 4. Show retrieval query
# --------------------------------------------------

query = build_dataset_semantic_query(
    dataset_evidence
)

print("\n" + "=" * 70)
print("DATASET RAG QUERY")
print("=" * 70)

print(query)


# --------------------------------------------------
# 5. Retrieve dataset-level knowledge
# --------------------------------------------------

rag_context = retrieve_dataset_context(
    evidence=dataset_evidence,
    retriever=retriever,
    top_k=8,
)

print("\n" + "=" * 70)
print("RETRIEVED DATASET KNOWLEDGE")
print("=" * 70)

for index, item in enumerate(
    rag_context,
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
# 6. Initialise LLM
# --------------------------------------------------

llm_provider = OllamaProvider(
    model="qwen2.5:3b"
)


# --------------------------------------------------
# 7. Dataset understanding
# --------------------------------------------------

result = understand_dataset(
    evidence=dataset_evidence,
    retriever=retriever,
    llm_provider=llm_provider,
    top_k=8,
)


# --------------------------------------------------
# 8. Show final semantic result
# --------------------------------------------------

print("\n" + "=" * 70)
print("DATASET SEMANTIC RESULT")
print("=" * 70)

pprint(
    result.model_dump()
)