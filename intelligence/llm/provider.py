from abc import ABC, abstractmethod
from typing import Any, Dict
from typing import Any, Dict, Type
from pydantic import BaseModel


class LLMProvider(ABC):
    """
    Abstract interface for any LLM used by LYNX.

    Pipeline components should depend on this interface,
    not directly on OpenAI, Gemini, Ollama, etc.
    """

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel]
    ) -> Dict[str, Any]:
        raise NotImplementedError
