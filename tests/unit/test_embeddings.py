from intelligence.rag.embeddings import (
    EmbeddingModel,
)


model = EmbeddingModel()

text = """
Category: semantic
Topic: identifier
Section: Important Distinction

Identifiers are values used to identify or reference
entities, records, or transactions.
"""


embedding = model.embed_text(
    text
)


print("\nEMBEDDING TEST")
print("=" * 60)

print(
    f"Model: {model.model_name}"
)

print(
    f"Vector dimensions: {len(embedding)}"
)

print(
    f"First 10 values: {embedding[:10]}"
)