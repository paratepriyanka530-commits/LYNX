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


def cosine_similarity(vector_a, vector_b):
    return sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )


model = EmbeddingModel()


# --------------------------------------------------
# Load semantic knowledge documents
# --------------------------------------------------

knowledge_files = [
    "knowledge_base/semantic/identifier.md",
    "knowledge_base/semantic/numeric_measurement.md",
]

all_chunks = []

for file_path in knowledge_files:

    document = load_knowledge_document(
        file_path
    )

    chunks = chunk_knowledge_document(
        document
    )

    all_chunks.extend(chunks)


print(
    f"\nTotal knowledge chunks loaded: {len(all_chunks)}"
)


# --------------------------------------------------
# Build embeddings for knowledge chunks
# --------------------------------------------------

chunk_embeddings = []

for chunk in all_chunks:

    embedding_text = build_embedding_text(
        chunk
    )

    embedding = model.embed_text(
        embedding_text
    )

    chunk_embeddings.append(
        {
            "chunk": chunk,
            "embedding": embedding,
        }
    )


# --------------------------------------------------
# Test query
# --------------------------------------------------

query = """
The values are stored as numbers, but they appear
to identify or reference transactions. Arithmetic
differences between the values are not meaningful.
"""

query_embedding = model.embed_text(
    query
)


# --------------------------------------------------
# Compare query with all knowledge chunks
# --------------------------------------------------

results = []

for item in chunk_embeddings:

    similarity = cosine_similarity(
        query_embedding,
        item["embedding"]
    )

    chunk = item["chunk"]

    results.append(
        {
            "chunk_id": chunk["chunk_id"],
            "topic": chunk["topic"],
            "section": chunk["section"],
            "score": similarity,
        }
    )


results.sort(
    key=lambda item: item["score"],
    reverse=True
)


# --------------------------------------------------
# Display ranking
# --------------------------------------------------

print("\nMULTI-DOCUMENT RETRIEVAL TEST")
print("=" * 80)

for result in results:

    print(
        f"{result['chunk_id']} | "
        f"{result['topic']:<20} | "
        f"{result['section']:<25} | "
        f"{result['score']:.4f}"
    )