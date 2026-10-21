from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from pydantic import BaseModel


class LLMProviderError(RuntimeError):
    """Base error for failures at the model-provider boundary."""


class LLMProviderUnavailableError(LLMProviderError):
    """The configured provider cannot currently be reached."""


class LLMProviderTimeoutError(LLMProviderError):
    """The provider did not return within its configured timeout."""


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


__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "LLMProviderTimeoutError",
    "LLMProviderUnavailableError",
]
