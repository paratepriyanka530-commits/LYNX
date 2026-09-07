from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Local embedding model used by LYNX RAG.

    Converts text into dense numerical vectors that can
    later be stored and compared by the vector store.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )


    def embed_text(
        self,
        text: str
    ) -> List[float]:
        """
        Generate an embedding for one text.
        """

        if not text or not text.strip():
            raise ValueError(
                "Cannot generate embedding for empty text."
            )

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()


    def embed_texts(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        """

        if not texts:
            raise ValueError(
                "No texts provided for embedding."
            )

        if any(
            not text or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "Embedding input contains empty text."
            )

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings.tolist()