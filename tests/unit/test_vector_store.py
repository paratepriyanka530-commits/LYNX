from intelligence.rag.knowledge_loader import (
    load_knowledge_document,
)

from intelligence.rag.knowledge_chunker import (
    chunk_knowledge_document,
)

from intelligence.rag.embedding_text_builder import (
    build_embedding_text,
)

from intelligence.rag.embeddings import (
    EmbeddingModel,
)

from intelligence.rag.vector_store import (
    InMemoryVectorStore,
)


model = EmbeddingModel()

store = InMemoryVectorStore()


knowledge_files = [
    "knowledge_base/semantic/identifier.md",
    "knowledge_base/semantic/numeric_measurement.md",
]


# ---------------------------------------------
# Index knowledge
# ---------------------------------------------

for file_path in knowledge_files:

    document = load_knowledge_document(
        file_path
    )

    chunks = chunk_knowledge_document(
        document
    )

    for chunk in chunks:

        text = build_embedding_text(
            chunk
        )

        embedding = model.embed_text(
            text
        )

        store.add(
            chunk=chunk,
            embedding=embedding
        )


print(
    f"\nStored knowledge chunks: {store.count()}"
)


# ---------------------------------------------
# Search
# ---------------------------------------------

query = """
The values are numeric but appear to identify
or reference transactions. Arithmetic differences
between the values are not meaningful.
"""


query_embedding = model.embed_text(
    query
)


results = store.search(
    query_embedding=query_embedding,
    top_k=5
)


print("\nVECTOR STORE SEARCH")
print("=" * 90)


for result in results:

    chunk = result["chunk"]

    print(
        f"{chunk['chunk_id']} | "
        f"{chunk['topic']:<20} | "
        f"{chunk['section']:<25} | "
        f"{result['score']:.4f}"
    )