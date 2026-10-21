import json
from typing import Any, Dict, Type

import requests
from pydantic import BaseModel

from intelligence.llm.provider import (
    LLMProvider,
    LLMProviderError,
    LLMProviderTimeoutError,
    LLMProviderUnavailableError,
)


class OllamaProvider(LLMProvider):

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        # Diagnostic metadata from the most recent call. This does not alter
        # the structured semantic result returned by generate_structured.
        self.last_response_metadata: Dict[str, Any] | None = None


    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel]
    ) -> Dict[str, Any]:

        self.last_response_metadata = None

        url = f"{self.base_url}/api/generate"

        schema = response_model.model_json_schema()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,

            # Real schema instead of only "json"
            "format": schema,

            "options": {
                "temperature": 0
            }
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=120
            )

            response.raise_for_status()

        except requests.Timeout as exc:
            raise LLMProviderTimeoutError("Ollama request timed out.") from exc
        except requests.ConnectionError as exc:
            raise LLMProviderUnavailableError("Ollama is unavailable.") from exc
        except requests.RequestException as exc:
            raise LLMProviderError("Ollama request failed.") from exc


        data = response.json()

        self.last_response_metadata = {
            key: data.get(key)
            for key in (
                "model",
                "created_at",
                "done",
                "done_reason",
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
        }

        raw_output = data.get("response")

        if not raw_output:
            raise RuntimeError(
                "Ollama returned an empty response."
            )


        try:
            validated = response_model.model_validate_json(
                raw_output
            )

        except Exception as exc:
            raise RuntimeError(
                "Ollama returned a response that does "
                "not satisfy the semantic schema.\n"
                f"Raw response:\n{raw_output}"
            ) from exc


        return validated.model_dump()
