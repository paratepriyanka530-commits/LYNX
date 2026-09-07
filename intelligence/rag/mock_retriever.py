from typing import Any, Dict, List

from intelligence.rag.retriever import RAGRetriever


class MockRAGRetriever(RAGRetriever):
    """
    Temporary retriever used during early development.

    It allows us to test the semantic pipeline before
    connecting a real vector store.
    """

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

        return [
            {
                "content": (
                    "Identifiers are values used to distinguish "
                    "or reference entities or records. Numeric "
                    "storage does not necessarily imply that "
                    "arithmetic operations are meaningful."
                ),
                "source": "mock_semantic_knowledge",
                "score": 1.0
            }
        ][:top_k]