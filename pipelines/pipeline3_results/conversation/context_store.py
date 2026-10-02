"""Replaceable, thread-safe in-memory conversation context store."""

from __future__ import annotations

from threading import RLock
from uuid import uuid4

from pipelines.pipeline3_results.conversation.schemas import (
    ConversationContext,
    ConversationQueryState,
    ConversationTurn,
    PendingClarification,
)


class ConversationNotFoundError(LookupError):
    pass


class ConversationDatasetMismatchError(LookupError):
    pass


class InMemoryConversationStore:
    """Stores immutable context snapshots; no dataframe or raw artifact is retained."""

    def __init__(self) -> None:
        self._contexts: dict[tuple[str, str], ConversationContext] = {}
        self._owners: dict[str, str] = {}
        self._lock = RLock()

    def create(self, dataset_id: str) -> ConversationContext:
        conversation_id = str(uuid4())
        context = ConversationContext(
            dataset_id=dataset_id, conversation_id=conversation_id
        )
        with self._lock:
            self._contexts[(dataset_id, conversation_id)] = context
            self._owners[conversation_id] = dataset_id
        return context

    def get(self, dataset_id: str, conversation_id: str) -> ConversationContext:
        with self._lock:
            owner = self._owners.get(conversation_id)
            if owner is None:
                raise ConversationNotFoundError(conversation_id)
            if owner != dataset_id:
                raise ConversationDatasetMismatchError(conversation_id)
            return self._contexts[(dataset_id, conversation_id)]

    def update(
        self,
        dataset_id: str,
        conversation_id: str,
        *,
        active_query: ConversationQueryState | None,
        pending_clarification: PendingClarification | None,
    ) -> ConversationContext:
        with self._lock:
            current = self.get(dataset_id, conversation_id)
            updated = current.model_copy(update={
                "active_query": active_query,
                "pending_clarification": pending_clarification,
            })
            self._contexts[(dataset_id, conversation_id)] = updated
            return updated

    def append_turn(
        self, dataset_id: str, conversation_id: str, turn: ConversationTurn
    ) -> ConversationContext:
        with self._lock:
            current = self.get(dataset_id, conversation_id)
            updated = current.model_copy(update={"turns": (*current.turns, turn)})
            self._contexts[(dataset_id, conversation_id)] = updated
            return updated

    def clear(self, dataset_id: str, conversation_id: str) -> None:
        with self._lock:
            self.get(dataset_id, conversation_id)
            del self._contexts[(dataset_id, conversation_id)]
            del self._owners[conversation_id]


__all__ = [
    "ConversationDatasetMismatchError",
    "ConversationNotFoundError",
    "InMemoryConversationStore",
]
