from intelligence.rag.knowledge_loader import (
    load_knowledge_document,
)

from intelligence.rag.knowledge_chunker import (
    chunk_knowledge_document,
)


document = load_knowledge_document(
    "knowledge_base/semantic/identifier.md"
)

chunks = chunk_knowledge_document(
    document
)


print("\nKNOWLEDGE CHUNKS")
print("=" * 60)

for chunk in chunks:

    print(f"\nChunk ID: {chunk['chunk_id']}")
    print(f"Topic: {chunk['topic']}")
    print(f"Section: {chunk['section']}")

    print("-" * 60)

    print(chunk["content"])

    print("=" * 60)


print(
    f"\nTotal chunks: {len(chunks)}"
)