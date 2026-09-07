from intelligence.rag.knowledge_loader import (
    load_knowledge_document,
)

from intelligence.rag.knowledge_chunker import (
    chunk_knowledge_document,
)

from intelligence.rag.embedding_text_builder import (
    build_embedding_text,
)


document = load_knowledge_document(
    "knowledge_base/semantic/identifier.md"
)

chunks = chunk_knowledge_document(
    document
)


print("\nEMBEDDING TEXT")
print("=" * 60)

for chunk in chunks:

    embedding_text = build_embedding_text(
        chunk
    )

    print(
        f"\nChunk: {chunk['chunk_id']}"
    )

    print("-" * 60)

    print(
        embedding_text
    )

    print("=" * 60)