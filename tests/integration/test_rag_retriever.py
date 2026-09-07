from intelligence.rag.retriever import (
    RAGRetriever,
)


retriever = RAGRetriever(
    knowledge_directory="knowledge_base/semantic"
)


print("\nINDEXING KNOWLEDGE...")
retriever.index_knowledge()

print(
    f"Indexed chunks: "
    f"{retriever.vector_store.count()}"
)


query = """
The values are numeric but appear to identify
or reference transactions. Arithmetic differences
between the values are not meaningful.
"""


results = retriever.retrieve(
    query=query,
    top_k=5
)


print("\nRAG RETRIEVAL RESULT")
print("=" * 90)


for result in results:

    print(
        f"{result['chunk_id']} | "
        f"{result['topic']:<20} | "
        f"{result['section']:<25} | "
        f"{result['score']:.4f}"
    )

    print(
        f"Source: {result['source']}"
    )

    print(
        result["content"][:250]
    )

    print("-" * 90)