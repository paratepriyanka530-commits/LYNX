"""Shared XAI pipeline for automatic and user-requested visualizations."""

from __future__ import annotations

import hashlib
import json
from threading import RLock

from intelligence.llm.provider import LLMProvider
from pipelines.pipeline3_results.execution.schemas import VerifiedResult
from pipelines.pipeline3_results.visualization.intelligence.schemas import (
    ChartFamily, RendererNeutralChartSpec, VisualizationPlan,
)
from .analyzer import analyze_chart, select_prose_facts
from .schemas import XAIFact, XAIResult
from .wording import generate_wording, selection_rationale


_ADVANCED = {
    ChartFamily.TREEMAP, ChartFamily.SUNBURST, ChartFamily.FUNNEL,
    ChartFamily.RADAR, ChartFamily.GAUGE, ChartFamily.SANKEY,
    ChartFamily.COMBINATION,
}


class VisualizationXAIService:
    """Build bounded prose from cached deterministic facts, never raw data."""

    _fact_cache: dict[str, tuple[tuple[XAIFact, ...], tuple[str, ...]]] = {}
    _cache_lock = RLock()

    @staticmethod
    def _key(result: VerifiedResult, plan: VisualizationPlan, spec: RendererNeutralChartSpec) -> str:
        payload = [
            result.model_dump(mode="json"), plan.model_dump(mode="json"),
            spec.model_dump(mode="json"),
        ]
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def build(
        self,
        result: VerifiedResult,
        plan: VisualizationPlan,
        spec: RendererNeutralChartSpec,
        *,
        provider: LLMProvider | None = None,
        language: str = "en",
    ) -> XAIResult:
        key = self._key(result, plan, spec)
        with self._cache_lock:
            cached = self._fact_cache.get(key)
        if cached is None:
            facts, limitations = analyze_chart(result, plan, spec)
            cached = (tuple(facts), tuple(dict.fromkeys(limitations)))
            with self._cache_lock:
                if len(self._fact_cache) >= 256:
                    self._fact_cache.pop(next(iter(self._fact_cache)))
                self._fact_cache[key] = cached
        facts, limitations = list(cached[0]), list(cached[1])
        selected = select_prose_facts(facts)
        why_chart, findings, final_limits, final_language, fallback = generate_wording(
            selected, plan, limitations, provider, language=language,
        )
        if result.status == "EMPTY":
            status = "EMPTY"
        elif spec.family in _ADVANCED or not [
            fact for fact in facts if fact.fact_type not in {"COVERAGE", "LIMITATION"}
        ]:
            status = "UNSUPPORTED"
        elif result.status == "PARTIAL":
            status = "PARTIAL"
        else:
            status = "READY"
        return XAIResult(
            visualization_id=plan.visualization_id, status=status,
            why_selected=selection_rationale(plan),
            why_this_chart=why_chart, what_it_shows=findings,
            facts=facts, limitations=final_limits, language=final_language,
            fallback_used=fallback,
        )


__all__ = ["VisualizationXAIService"]
