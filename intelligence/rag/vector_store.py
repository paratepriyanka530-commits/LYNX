from typing import Any, Dict, List


class InMemoryVectorStore:
    """
    Simple in-memory vector store for LYNX RAG.

    Stores knowledge chunks together with their embeddings
    and performs cosine-similarity search.

    Embeddings are expected to already be normalized.
    """

    def __init__(self):
        self._records: List[Dict[str, Any]] = []


    def add(
        self,
        chunk: Dict[str, Any],
        embedding: List[float]
    ) -> None:
        """
        Store one knowledge chunk and its embedding.
        """

        if not embedding:
            raise ValueError(
                "Embedding cannot be empty."
            )

        self._records.append(
            {
                "chunk": chunk,
                "embedding": embedding,
            }
        )


    def add_many(
        self,
        records: List[Dict[str, Any]]
    ) -> None:
        """
        Store multiple chunk/embedding records.

        Expected record format:

        {
            "chunk": {...},
            "embedding": [...]
        }
        """

        for record in records:

            if "chunk" not in record:
                raise ValueError(
                    "Vector-store record is missing 'chunk'."
                )

            if "embedding" not in record:
                raise ValueError(
                    "Vector-store record is missing 'embedding'."
                )

            self.add(
                chunk=record["chunk"],
                embedding=record["embedding"]
            )


    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        categories: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the most similar knowledge chunks.

        If categories are provided, only chunks belonging
        to those knowledge categories are searched.
        """

        if not query_embedding:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        results = []

        for record in self._records:

            chunk = record["chunk"]

            if (
                categories is not None
                and chunk.get("category") not in categories
            ):
                continue

            score = self._cosine_similarity(
                query_embedding,
                record["embedding"]
            )

            results.append(
                {
                    "chunk": chunk,
                    "score": score,
                }
            )

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]


    def count(self) -> int:
        return len(self._records)


    @staticmethod
    def _cosine_similarity(
        vector_a: List[float],
        vector_b: List[float]
    ) -> float:
        """
        Because our embedding model produces normalized
        vectors, their dot product equals cosine similarity.
        """

        if len(vector_a) != len(vector_b):
            raise ValueError(
                "Embedding dimensions do not match."
            )

        return sum(
            a * b
            for a, b in zip(
                vector_a,
                vector_b
            )
        )