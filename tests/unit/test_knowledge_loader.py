from pprint import pprint

from intelligence.rag.knowledge_loader import (
    load_knowledge_document,
)


file_path = (
    "knowledge_base/semantic/identifier.md"
)


document = load_knowledge_document(
    file_path
)


print("\nKNOWLEDGE DOCUMENT")
print("=" * 60)

pprint(document)