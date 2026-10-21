"""Cached composition service for the Insights dataset landing state."""

from __future__ import annotations

import hashlib
import json
from threading import RLock
from typing import Any, Mapping

import pandas as pd

from intelligence.llm.provider import LLMProvider
from pipelines.pipeline3_results.grounding.dataset_knowledge import build_dataset_knowledge
from pipelines.pipeline3_results.insights.landing.schemas import (
    InsightsLandingState,
    SuggestedQuestionItem,
    SummaryPackage,
)
from pipelines.pipeline3_results.insights.landing.suggestions import (
    generate_suggested_questions,
)
from pipelines.pipeline3_results.insights.landing.summary_engine import (
    build_summary_package,
)
from pipelines.pipeline3_results.insights.landing.wording import SummaryWordingService


def _version(
    dataframe: pd.DataFrame,
    pipeline1_report: Mapping[str, Any] | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(dataframe.shape).encode("utf-8"))
    digest.update("|".join(map(str, dataframe.columns)).encode("utf-8"))
    digest.update("|".join(map(str, dataframe.dtypes)).encode("utf-8"))
    try:
        content_hash = pd.util.hash_pandas_object(dataframe, index=True)
    except TypeError:
        content_hash = pd.util.hash_pandas_object(dataframe.astype(str), index=True)
    digest.update(content_hash.values.tobytes())
    if pipeline1_report:
        digest.update(json.dumps(
            pipeline1_report, sort_keys=True, default=str, separators=(",", ":")
        ).encode("utf-8"))
    return digest.hexdigest()[:20]


class InsightsLandingService:
    """Cache deterministic facts separately from provider-generated wording."""

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider
        self._fact_cache: dict[tuple[str, str], SummaryPackage] = {}
        self._landing_cache: dict[tuple[str, str, str, str], InsightsLandingState] = {}
        self._lock = RLock()

    def build(
        self,
        dataset_id: str,
        dataframe: pd.DataFrame,
        *,
        pipeline1_report: Mapping[str, Any] | None = None,
        dataset_name: str | None = None,
        requested_language: str = "en",
        data_version: str | None = None,
        provider: LLMProvider | None = None,
    ) -> InsightsLandingState:
        active_provider = provider if provider is not None else self.provider
        version = data_version or _version(dataframe, pipeline1_report)
        provider_key = type(active_provider).__qualname__ if active_provider else "none"
        landing_key = (dataset_id, version, requested_language, provider_key)
        with self._lock:
            cached = self._landing_cache.get(landing_key)
        if cached is not None:
            return cached.model_copy(update={
                "signals": [*cached.signals, "insights_landing:cache_hit"],
            })

        knowledge = build_dataset_knowledge(
            pipeline1_report,
            dataset_id=dataset_id,
            dataframe_schema={str(name): str(dtype) for name, dtype in dataframe.dtypes.items()},
        )
        fact_key = (dataset_id, version)
        with self._lock:
            package = self._fact_cache.get(fact_key)
        fact_cache_hit = package is not None
        if package is None:
            package = build_summary_package(
                dataframe, knowledge, dataset_id=dataset_id,
            )
            with self._lock:
                self._fact_cache[fact_key] = package

        wording = SummaryWordingService().generate(
            package, active_provider, requested_language=requested_language,
        )
        questions = generate_suggested_questions(
            knowledge, requested_language=requested_language,
        )
        state = InsightsLandingState(
            dataset=dataset_name or dataset_id,
            status=package.status,
            summary_text=wording.text,
            summary=package,
            suggested_questions=[SuggestedQuestionItem(
                id=item.id, text=item.text, analytical_goal=item.analytical_goal,
            ) for item in questions],
            signals=[
                "insights_landing:deterministic_facts",
                "insights_landing:answerable_suggestions",
                (
                    "insights_landing:summary_fallback"
                    if wording.fallback_used
                    else "insights_landing:summary_model_validated"
                ),
                (
                    "insights_landing:fact_cache_hit"
                    if fact_cache_hit else "insights_landing:fact_cache_miss"
                ),
            ],
        )
        with self._lock:
            self._landing_cache[landing_key] = state
        return state


__all__ = ["InsightsLandingService"]
