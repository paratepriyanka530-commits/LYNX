from intelligence.rag.embeddings import EmbeddingModel


def cosine_similarity(
    vector_a,
    vector_b
):
    return sum(
        a * b
        for a, b in zip(
            vector_a,
            vector_b
        )
    )


model = EmbeddingModel()


query = """
A numeric column that is used to identify
or reference transactions rather than measure quantity.
"""


texts = {
    "identifier": """
    Identifiers are values used to distinguish,
    reference, or link entities, records,
    transactions, or events.
    """,

    "measurement": """
    Numeric measurements represent quantities
    where arithmetic magnitude is meaningful,
    such as temperature, weight, price,
    age, or distance.
    """,

    "date": """
    Date values represent calendar dates
    associated with records or events.
    """
}


query_embedding = model.embed_text(
    query
)


print("\nEMBEDDING SIMILARITY TEST")
print("=" * 60)


for label, text in texts.items():

    text_embedding = model.embed_text(
        text
    )

    similarity = cosine_similarity(
        query_embedding,
        text_embedding
    )

    print(
        f"{label:15} -> {similarity:.4f}"
    )