"""Deterministic ambiguity decisions over Phase 4 column candidates.

The detector treats Phase 4 scores as ordinal evidence strengths, never as
probabilities.  A candidate is viable only when it has substantive name,
alias, or Pipeline 1 semantic evidence; analytical-role support alone cannot
manufacture one.

Clear-winner rule
-----------------
* A sole viable candidate is resolved, even when supported only by a weak but
  legitimate alias or semantic signal.
* With multiple candidates, the winner is resolved only when its strongest
  evidence tier is strictly higher than every competitor's tier.
* Candidates in the same strongest tier are near-equal.  They require
  clarification even if the optional role-support bonus made their numeric
  scores differ slightly.  Column order never chooses semantic meaning.

Only candidates in the strongest competing tier appear as clarification
choices.  Their order follows Phase 4's deterministic score/order output.
"""

from __future__ import annotations

from collections.abc import Sequence

from pipelines.pipeline3_results.nlp.schemas import (
    AmbiguityDecision,
    AmbiguityResult,
    ColumnCandidate,
    TermResolution,
)


_EXACT_NAME_TIER = 4
_TOKEN_TIER = 3
_SEMANTIC_TIER = 2
_ALIAS_TIER = 1


def _evidence_tier(candidate: ColumnCandidate) -> int:
    """Return the strongest substantive Phase 4 evidence tier, or zero."""
    tier = 0
    for source in candidate.evidence_sources:
        if source == "exact_normalized_name":
            tier = max(tier, _EXACT_NAME_TIER)
        elif source == "semantic_entity_name_match":
            tier = max(tier, _EXACT_NAME_TIER)
        elif source == "semantic_count_identifier_match":
            tier = max(tier, _EXACT_NAME_TIER)
        elif source == "exact_token_match":
            tier = max(tier, _TOKEN_TIER)
        elif source in {
            "semantic_description_match",
            "semantic_metadata_match",
            "semantic_role_match",
        } or source.startswith("semantic_concept_match:"):
            tier = max(tier, _SEMANTIC_TIER)
        elif source.startswith("alias_match:") or source.startswith(
            "semantic_alias_match:"
        ) or source.startswith("semantic_concept_candidate:"):
            tier = max(tier, _ALIAS_TIER)
    return tier


def _ordered_viable_candidates(
    resolution: TermResolution,
) -> list[tuple[int, int, ColumnCandidate]]:
    """Filter role-only candidates and preserve Phase 4 ordering for ties."""
    viable = [
        (index, _evidence_tier(candidate), candidate)
        for index, candidate in enumerate(resolution.candidates)
        if _evidence_tier(candidate) > 0
    ]
    return sorted(
        viable,
        key=lambda item: (
            -item[1],
            -(item[2].score if item[2].score is not None else 0.0),
            item[0],
            item[2].column_name.casefold(),
        ),
    )


def detect_ambiguity(
    resolution: TermResolution,
    *,
    required: bool = True,
) -> AmbiguityDecision:
    """Decide one term without inspecting values or changing its candidates."""
    if not isinstance(resolution, TermResolution):
        raise TypeError("resolution must be a TermResolution")

    viable = _ordered_viable_candidates(resolution)
    base_signals = (f"term_kind:{resolution.kind}",)

    if not viable:
        if not required:
            return AmbiguityDecision(
                term=resolution.term,
                kind=resolution.kind,
                status="RESOLVED",
                required=False,
                reason="optional_term_unsupported",
                signals=base_signals + ("optional_term_skipped",),
            )
        return AmbiguityDecision(
            term=resolution.term,
            kind=resolution.kind,
            status="INVALID",
            required=True,
            reason="no_supported_candidate",
            signals=base_signals + ("no_supported_candidate",),
        )

    top_tier = viable[0][1]
    top_tier_candidates = [item for item in viable if item[1] == top_tier]

    if len(top_tier_candidates) == 1:
        winner = top_tier_candidates[0][2]
        if len(viable) == 1:
            if top_tier == _EXACT_NAME_TIER:
                reason = "clear_exact_match"
                decision_signals = (
                    "single_supported_candidate",
                    "clear_exact_match",
                )
            else:
                reason = "sole_supported_candidate"
                decision_signals = ("single_supported_candidate",)
        elif top_tier == _EXACT_NAME_TIER:
            reason = "clear_exact_match"
            decision_signals = ("clear_exact_match",)
        else:
            reason = "clear_evidence_tier_separation"
            decision_signals = ("clear_tier_separation",)
        return AmbiguityDecision(
            term=resolution.term,
            kind=resolution.kind,
            status="RESOLVED",
            required=required,
            resolved_column=winner.column_name,
            reason=reason,
            signals=base_signals + decision_signals,
        )

    scores = [item[2].score for item in top_tier_candidates]
    equal_scores = all(score == scores[0] for score in scores[1:])
    ambiguity_signal = (
        "equal_top_candidates" if equal_scores else "near_equal_candidates"
    )
    return AmbiguityDecision(
        term=resolution.term,
        kind=resolution.kind,
        status="NEEDS_CLARIFICATION",
        required=required,
        choices=[item[2].column_name for item in top_tier_candidates],
        reason="multiple_near_equal_candidates",
        signals=base_signals + (ambiguity_signal,),
    )


def detect_ambiguities(
    resolutions: Sequence[TermResolution],
    *,
    required: Sequence[bool] | None = None,
) -> AmbiguityResult:
    """Evaluate several terms using INVALID > clarification > resolved."""
    if isinstance(resolutions, (str, bytes)) or not isinstance(resolutions, Sequence):
        raise TypeError("resolutions must be a sequence of TermResolution values")
    if required is None:
        requirements = [True] * len(resolutions)
    else:
        requirements = list(required)
        if len(requirements) != len(resolutions):
            raise ValueError("required must have one flag per resolution")
        if any(not isinstance(flag, bool) for flag in requirements):
            raise TypeError("required flags must be booleans")

    decisions = [
        detect_ambiguity(resolution, required=is_required)
        for resolution, is_required in zip(resolutions, requirements, strict=True)
    ]
    required_decisions = [decision for decision in decisions if decision.required]
    signals: list[str] = []
    if len(required_decisions) > 1:
        signals.append("multiple_required_terms")

    if any(decision.status == "INVALID" for decision in required_decisions):
        status = "INVALID"
        reason = "one_or_more_required_terms_invalid"
        signals.append("overall_invalid")
    elif any(
        decision.status == "NEEDS_CLARIFICATION" for decision in required_decisions
    ):
        status = "NEEDS_CLARIFICATION"
        reason = "one_or_more_required_terms_ambiguous"
        signals.append("overall_needs_clarification")
    else:
        status = "RESOLVED"
        reason = "all_required_terms_resolved"
        signals.append("overall_resolved")

    return AmbiguityResult(
        status=status,
        decisions=decisions,
        reason=reason,
        signals=tuple(signals),
    )


__all__ = ["detect_ambiguities", "detect_ambiguity"]
