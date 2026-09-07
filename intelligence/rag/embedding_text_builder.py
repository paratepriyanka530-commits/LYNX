from typing import Any, Dict


def build_embedding_text(
    chunk: Dict[str, Any]
) -> str:
    """
    Build the text representation that will be embedded
    for semantic retrieval.
    """

    return (
        f"Category: {chunk['category']}\n"
        f"Topic: {chunk['topic']}\n"
        f"Section: {chunk['section']}\n\n"
        f"{chunk['content']}"
    )