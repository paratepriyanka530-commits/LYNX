"""Provider-neutral generation of wording from verified analytical facts."""

from __future__ import annotations

from pydantic import ValidationError

from intelligence.llm.provider import (
    LLMProvider,
    LLMProviderError,
    LLMProviderTimeoutError,
    LLMProviderUnavailableError,
)
from pipelines.pipeline3_results.execution.schemas import VerifiedResult
from pipelines.pipeline3_results.insights.response_generation.fallback import (
    deterministic_fallback,
)
from pipelines.pipeline3_results.insights.response_generation.prompt import (
    build_response_prompt,
)
from pipelines.pipeline3_results.insights.response_generation.schemas import (
    ModelResponseDraft,
    NaturalLanguageResponse,
    ResponseContext,
)
from pipelines.pipeline3_results.insights.response_generation.validator import (
    validate_response,
)
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan


class InsightsResponseGenerator:
    """Explain a VerifiedResult without performing analytical computation."""

    def generate(
        self,
        response_context: ResponseContext,
        provider: LLMProvider | None,
        *,
        plan: AnalyticalPlan,
        verified_result: VerifiedResult,
    ) -> NaturalLanguageResponse:
        if provider is None or verified_result.status not in {"READY", "PARTIAL"}:
            return deterministic_fallback(response_context, plan, verified_result)
        try:
            raw = provider.generate_structured(
                build_response_prompt(response_context), ModelResponseDraft
            )
            draft = ModelResponseDraft.model_validate(raw)
            validation = validate_response(draft, response_context)
            if not validation.valid:
                return deterministic_fallback(response_context, plan, verified_result)
            caveats = list(response_context.warnings)
            return NaturalLanguageResponse(
                text=draft.text,
                language=draft.language,
                status=("PARTIAL" if verified_result.status == "PARTIAL" else "READY"),
                verified_result_id=verified_result.result_id,
                caveats=caveats,
                fallback_used=False,
            )
        except (
            LLMProviderUnavailableError,
            LLMProviderTimeoutError,
            LLMProviderError,
            TimeoutError,
            ValidationError,
            TypeError,
            ValueError,
        ):
            return deterministic_fallback(response_context, plan, verified_result)
        except Exception:
            # Provider implementations are external boundaries; do not leak details.
            return deterministic_fallback(response_context, plan, verified_result)


__all__ = ["InsightsResponseGenerator"]
