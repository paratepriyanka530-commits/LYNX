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

document = load_knowledge_document(
    "knowledge_base/semantic/identifier.md"
)

chunks = chunk_knowledge_document(
    document
)


query = """
The values are stored as numbers, but they appear to be used
as labels or references for transactions. Arithmetic differences
between the values do not appear meaningful.
"""


query_embedding = model.embed_text(
    query
)


results = []

for chunk in chunks:

    embedding_text = build_embedding_text(
        chunk
    )

    chunk_embedding = model.embed_text(
        embedding_text
    )

    similarity = cosine_similarity(
        query_embedding,
        chunk_embedding
    )

    results.append(
        {
            "chunk_id": chunk["chunk_id"],
            "section": chunk["section"],
            "score": similarity,
        }
    )


results.sort(
    key=lambda item: item["score"],
    reverse=True
)


print("\nCHUNK RETRIEVAL TEST")
print("=" * 60)

for result in results:

    print(
        f"{result['chunk_id']} | "
        f"{result['section']:<25} | "
        f"{result['score']:.4f}"
    )