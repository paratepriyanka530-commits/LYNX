"""Model-agnostic semantic similarity boundary for grounding."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher


def semantic_tokens(text: str) -> list[str]:
    normalized = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    normalized = re.sub(r"[_\W]+", " ", normalized.casefold())
    tokens: list[str] = []
    for token in normalized.split():
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        tokens.append(token)
    return tokens


class SemanticSimilarityProvider(ABC):
    """Provider-neutral similarity interface; no dataset rows are accepted."""

    @abstractmethod
    def similarity(self, query_text: str, candidate_text: str) -> float:
        raise NotImplementedError


class LexicalSemanticSimilarityProvider(SemanticSimilarityProvider):
    """Deterministic metadata-only fallback with no synonym dictionary."""

    def similarity(self, query_text: str, candidate_text: str) -> float:
        query_tokens = set(semantic_tokens(query_text))
        candidate_tokens = set(semantic_tokens(candidate_text))
        if not query_tokens or not candidate_tokens:
            return 0.0
        if query_tokens == candidate_tokens:
            return 1.0
        overlap = len(query_tokens & candidate_tokens)
        coverage = overlap / len(query_tokens)
        precision = overlap / len(candidate_tokens)
        sequence = SequenceMatcher(
            None,
            " ".join(sorted(query_tokens)),
            " ".join(sorted(candidate_tokens)),
        ).ratio()
        return round(min(1.0, 0.75 * coverage + 0.15 * precision + 0.10 * sequence), 6)


__all__ = [
    "LexicalSemanticSimilarityProvider",
    "SemanticSimilarityProvider",
    "semantic_tokens",
]
