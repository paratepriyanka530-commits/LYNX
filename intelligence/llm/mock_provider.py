from typing import Any, Dict
from typing import Any, Dict, Type
from pydantic import BaseModel

from intelligence.llm.provider import (
    LLMProvider,
)


class MockLLMProvider(LLMProvider):
    """
    Temporary deterministic LLM provider.

    Used only to test the semantic pipeline
    before connecting a real language model.
    """

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel]
    ) -> Dict[str, Any]:

        return {
            "semantic_type": "identifier",

            "semantic_role": (
                "transaction_identifier"
            ),

            "description": (
                "Identifier associated with "
                "a transaction or order."
            ),

            "confidence": 0.95,

            "reasoning_summary": (
                "The column appears to represent "
                "an identifier that may repeat because "
                "multiple records can belong to the "
                "same transaction."
            ),

            "possible_issues": []
        }