from pathlib import Path
from typing import Any, Dict, List

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

from intelligence.rag.vector_store import (
    InMemoryVectorStore,
)


class RAGRetriever:
    """
    Local RAG retriever for LYNX.

    Loads knowledge documents, creates embeddings,
    indexes them in a vector store, and retrieves
    relevant knowledge for a query.
    """

    def __init__(
        self,
        knowledge_directory: str,
        embedding_model: EmbeddingModel | None = None,
        vector_store: InMemoryVectorStore | None = None,
    ):
        self.knowledge_directory = Path(
            knowledge_directory
        )

        self.embedding_model = (
            embedding_model
            or EmbeddingModel()
        )

        self.vector_store = (
            vector_store
            or InMemoryVectorStore()
        )

        self._indexed = False


    def index_knowledge(self) -> None:
        """
        Load and index all Markdown knowledge documents
        inside the configured directory.
        """

        if not self.knowledge_directory.exists():
            raise FileNotFoundError(
                "Knowledge directory not found: "
                f"{self.knowledge_directory}"
            )

        if not self.knowledge_directory.is_dir():
            raise ValueError(
                "Knowledge path is not a directory: "
                f"{self.knowledge_directory}"
            )


        knowledge_files = sorted(
            file_path
            for file_path in self.knowledge_directory.rglob("*.md")
            if file_path.name not in {
                "README.md",
                "MANIFEST.md",
                "KNOWLEDGE_SCHEMA.md",
            }
        )

        if not knowledge_files:
            raise ValueError(
                "No Markdown knowledge documents found in "
                f"{self.knowledge_directory}"
            )


        for file_path in knowledge_files:

            document = load_knowledge_document(
                str(file_path)
            )

            chunks = chunk_knowledge_document(
                document
            )

            embedding_texts = [
                build_embedding_text(chunk)
                for chunk in chunks
            ]

            embeddings = (
                self.embedding_model.embed_texts(
                    embedding_texts
                )
            )

            for chunk, embedding in zip(
                chunks,
                embeddings
            ):
                self.vector_store.add(
                    chunk=chunk,
                    embedding=embedding
                )


        self._indexed = True


    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        categories: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant knowledge chunks.
        """

        if not query or not query.strip():
            raise ValueError(
                "RAG query cannot be empty."
            )

        if not self._indexed:
            raise RuntimeError(
                "Knowledge has not been indexed. "
                "Call index_knowledge() first."
            )


        query_embedding = (
            self.embedding_model.embed_text(
                query
            )
        )


        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            categories=categories,
        )


        formatted_results = []

        for result in results:

            chunk = result["chunk"]

            formatted_results.append(
                {
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "score": result["score"],
                    "knowledge_id": chunk["knowledge_id"],
                    "chunk_id": chunk["chunk_id"],
                    "category": chunk["category"],
                    "topic": chunk["topic"],
                    "section": chunk["section"],
                }
            )


        return formatted_results