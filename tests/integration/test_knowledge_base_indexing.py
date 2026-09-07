from pathlib import Path

from intelligence.rag.knowledge_loader import (
    load_knowledge_document,
)

from intelligence.rag.knowledge_chunker import (
    chunk_knowledge_document,
)


KNOWLEDGE_BASE_PATH = Path("knowledge_base")


def main():

    knowledge_files = sorted(
        file_path
        for file_path in KNOWLEDGE_BASE_PATH.rglob("*.md")
        if file_path.name not in {
            "README.md",
            "MANIFEST.md",
            "KNOWLEDGE_SCHEMA.md",
        }
    )

    print("\nLYNX KNOWLEDGE BASE INDEXING TEST")
    print("=" * 50)

    print(f"\nKnowledge documents found: {len(knowledge_files)}")

    total_chunks = 0
    categories = set()

    for file_path in knowledge_files:

        document = load_knowledge_document(
            str(file_path)
        )

        chunks = chunk_knowledge_document(
            document
        )

        total_chunks += len(chunks)

        categories.add(
            document["category"]
        )

    print(f"Total chunks created: {total_chunks}")

    print("\nCategories found:")

    for category in sorted(categories):
        print(f"  - {category}")

    print("\nExample files:")

    for file_path in knowledge_files[:10]:
        print(f"  - {file_path}")

    print("\nKnowledge base parsing test completed successfully.")


if __name__ == "__main__":
    main()