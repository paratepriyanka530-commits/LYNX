"""Hybrid conservative Cleaning Decision Engine."""

from __future__ import annotations

import time
import json
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, Sequence, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from intelligence.llm.provider import LLMProvider
from pipelines.pipeline1_cleaning.decisions.decision_schema import CleaningDecision
from pipelines.pipeline1_cleaning.quality.quality_report import DetectedIssue
from pipelines.pipeline1_cleaning.semantics.batched_schema import SemanticProfile
from pipelines.pipeline1_cleaning.techniques.numeric_coercion import QUANTITATIVE_TYPES
from pipelines.pipeline1_cleaning.techniques.missing_values import (
    assess_median_semantic_eligibility,
    missingness_assessment,
    prepare_imputation,
)


class Tier2DecisionPayload(BaseModel):
    """Compact model-authored portion of a final CleaningDecision."""

    issue_id: str = Field(min_length=1)
    decision_type: Literal[
        "action", "no_action", "flag_for_review", "insufficient_evidence",
    ]
    proposed_action: Literal[
        "no_op", "trim_whitespace", "normalize_categorical_labels",
        "parse_dates", "coerce_dtype", "impute_missing",
        "add_missing_indicator", "drop_duplicate_rows", "flag_rows",
    ]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=240)


class Tier2DecisionBatch(BaseModel):
    decisions: List[Tier2DecisionPayload]


class _StrictMissingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyMissingParameters(_StrictMissingModel):
    pass


class MedianMissingParameters(_StrictMissingModel):
    strategy: Literal["median"]


class Tier2MissingActionPayload(_StrictMissingModel):
    issue_id: str = Field(min_length=1)
    decision_type: Literal["action"]
    proposed_action: Literal["impute_missing"]
    parameters: MedianMissingParameters
    recommendation_basis: Literal["semantically_appropriate_imputation"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=240)


class Tier2MissingNoActionPayload(_StrictMissingModel):
    issue_id: str = Field(min_length=1)
    decision_type: Literal["no_action"]
    proposed_action: Literal["no_op"]
    parameters: EmptyMissingParameters
    recommendation_basis: Literal["affirmative_preservation_evidence"]
    preservation_evidence_keys: List[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=240)


class Tier2MissingInsufficientEvidencePayload(_StrictMissingModel):
    issue_id: str = Field(min_length=1)
    decision_type: Literal["insufficient_evidence"]
    proposed_action: Literal["no_op"]
    parameters: EmptyMissingParameters
    recommendation_basis: Literal["insufficient_or_contradictory_evidence"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=240)


class Tier2MissingFlagForReviewPayload(_StrictMissingModel):
    issue_id: str = Field(min_length=1)
    decision_type: Literal["flag_for_review"]
    proposed_action: Literal["flag_rows"]
    parameters: EmptyMissingParameters
    review_basis: Literal[
        "protected_entity_reference",
        "temporal_value_not_safely_synthesizable",
        "high_missingness_requires_review",
        "other_supplied_consequence",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=240)


Tier2MissingDecisionPayload = Annotated[
    Union[
        Tier2MissingActionPayload,
        Tier2MissingNoActionPayload,
        Tier2MissingInsufficientEvidencePayload,
        Tier2MissingFlagForReviewPayload,
    ],
    Field(discriminator="decision_type"),
]


class Tier2MissingDecisionBatch(_StrictMissingModel):
    decisions: List[Tier2MissingDecisionPayload]


@dataclass(frozen=True)
class DecisionConfig:
    max_issues_per_batch: int = 4
    retry_count: int = 1


def _decision_id(issue_id: str) -> str:
    return f"decision-{issue_id}"


def _semantic_for(issue: DetectedIssue, profile: SemanticProfile) -> Dict[str, Any]:
    indexed = {item.column_name: item for item in profile.columns}
    return {
        name: indexed[name].model_dump(mode="json")
        for name in issue.target_columns if name in indexed
    }


def _tier1(issue: DetectedIssue, semantic: Dict[str, Any]) -> CleaningDecision | None:
    common = dict(
        decision_id=_decision_id(issue.issue_id), source_issue_ids=[issue.issue_id],
        target_columns=issue.target_columns, evidence_summary=issue.evidence,
        semantic_basis=semantic,
    )
    if issue.issue_type == "whitespace_issue":
        return CleaningDecision(
            **common, decision_type="action", proposed_action="trim_whitespace",
            confidence=0.98, rationale="Observed leading, trailing, or whitespace-only values can be trimmed deterministically.",
            risk_level="low", requires_review=False, decision_source="deterministic_tier1",
        )
    if issue.issue_type == "rare_category_observation":
        return CleaningDecision(
            **common, decision_type="no_action", proposed_action="no_op",
            confidence=0.99, rationale="Rarity alone does not establish that a category is invalid.",
            risk_level="low", requires_review=False, decision_source="deterministic_tier1",
        )
    if issue.issue_type == "statistical_outlier_candidate":
        return CleaningDecision(
            **common, decision_type="flag_for_review", proposed_action="flag_rows",
            confidence=0.99, rationale="A statistical outlier may be valid and must not be clipped or removed automatically.",
            risk_level="high", requires_review=True, decision_source="deterministic_tier1",
        )
    if issue.issue_type == "exact_duplicate_rows":
        return CleaningDecision(
            **common, decision_type="flag_for_review", proposed_action="flag_rows",
            confidence=0.99, rationale="Exact duplicate records may be legitimate repeated events; deletion is not automatic.",
            risk_level="high", requires_review=True, decision_source="deterministic_tier1",
        )
    if issue.issue_type == "missing_values":
        functions = {value.get("value_function") for value in semantic.values()}
        if "entity_reference" in functions:
            return CleaningDecision(
                **common, decision_type="flag_for_review", proposed_action="flag_rows",
                confidence=0.95, rationale="Missing entity references must not be fabricated or automatically imputed.",
                risk_level="high", requires_review=True, decision_source="deterministic_tier1",
            )
        if not semantic or functions <= {"unknown"}:
            return CleaningDecision(
                **common, decision_type="insufficient_evidence", proposed_action="no_op",
                confidence=0.9, rationale="Semantic meaning is insufficient to select a safe missing-value action.",
                risk_level="high", requires_review=True, decision_source="fallback",
            )
    return None


def _prompt(issues: Sequence[DetectedIssue], profile: SemanticProfile) -> str:
    payload = []
    for issue in issues:
        semantics = _semantic_for(issue, profile)
        compact_semantics = {
            column: {
                key: values.get(key)
                for key in (
                    "semantic_type", "value_function", "concept_description", "confidence",
                )
            }
            for column, values in semantics.items()
        }
        item = {
            "issue_id": issue.issue_id,
            "issue_type": issue.issue_type,
            "target_columns": issue.target_columns,
            "affected_count": issue.affected_count,
            "affected_percentage": issue.affected_percentage,
            "observed_evidence": issue.evidence,
            "column_semantics": compact_semantics,
        }
        if issue.examples:
            item["examples"] = issue.examples[:3]
        payload.append(item)
    context = profile.dataset_context
    return f"""
Return exactly one compact decision for each issue. Copy its exact issue_id;
do not omit, duplicate, or invent IDs. Keep rationale to one short sentence.

Allowed actions: no_op, trim_whitespace,
normalize_categorical_labels, parse_dates, coerce_dtype, impute_missing,
add_missing_indicator, drop_duplicate_rows, flag_rows.

Policy: preserve data when uncertain; never fabricate entity references;
do not automatically remove outliers/duplicates or invalidate rare categories.
impute_missing requires supported measurement semantics and non-extreme
missingness; parse_dates requires temporal semantics; coerce_dtype requires
measurement semantics and strong parseability. Return structured data only.
Safety, lineage, and review policy are attached deterministically after this call.

Shared dataset context: row_entity={context.row_entity!r}, probable_domain={context.probable_domain!r}

Issues:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)}
""".strip()


def _missing_value_readiness(
    issue: DetectedIssue,
    profile: SemanticProfile,
    dataframe: pd.DataFrame | None,
) -> Dict[str, Any]:
    """Build redacted advisory facts; downstream validation remains authoritative."""
    semantic = next(iter(_semantic_for(issue, profile).values()), {})
    column = issue.target_columns[0] if len(issue.target_columns) == 1 else None
    preservation_keys = (
        "missingness_is_meaningful",
        "approved_missingness_interpretation",
        "missingness_justification",
    )
    preservation_evidence = {
        key: issue.evidence[key] for key in preservation_keys if key in issue.evidence
    }
    supplied_consequence = issue.evidence.get("missingness_consequence_evidence")
    affirmative_preservation = (
        issue.evidence.get("missingness_is_meaningful") is True
        or issue.evidence.get("approved_missingness_interpretation") is not None
    )

    if dataframe is None or column is None or column not in dataframe.columns:
        return {
            "missing_count": issue.affected_count,
            "missing_percentage": issue.affected_percentage,
            "missingness_band": "unknown",
            "observed_count": issue.evidence.get("non_null_count"),
            "physical_dtype": None,
            "semantic_type": semantic.get("semantic_type"),
            "value_function": semantic.get("value_function"),
            "numeric_meaning": semantic.get("numeric_meaning"),
            "semantic_confidence": semantic.get("confidence"),
            "concept_description": semantic.get("concept_description"),
            "identifier_or_entity_reference_protected": None,
            "affirmative_preservation_evidence": (
                preservation_evidence if affirmative_preservation else None
            ),
            "supplied_consequence_evidence": supplied_consequence,
            "median_semantic_eligibility": {
                "status": "unknown", "reason_code": "observed_values_unavailable",
            },
            "median_mechanical_preflight": {
                "status": "unknown", "reason_code": "observed_values_unavailable",
            },
        }

    series = dataframe[column]
    missingness = missingness_assessment(series)
    semantic_result = assess_median_semantic_eligibility(
        semantic, series, issue.evidence,
    )
    try:
        # This is a read-only advisory preflight. Deliberately discard the
        # calculated replacement; authorization/compiler validation repeats it.
        prepare_imputation(series, {"strategy": "median"})
        mechanical = {
            "status": "pass", "reason_code": "existing_imputation_contract_passed",
        }
    except (ValueError, TypeError) as exc:
        mechanical = {
            "status": "fail", "reason_code": "existing_imputation_contract_failed",
            "reason": str(exc),
        }

    semantic_reason = semantic_result["reason_code"]
    protected = semantic_reason in {
        "identifier_or_entity_reference", "numeric_identifier",
    }
    readiness = {
        "missing_count": missingness["missing_count"],
        "missing_percentage": missingness["missing_percentage"],
        "missingness_band": missingness["missingness_risk"],
        "observed_count": missingness["observed_count"],
        "physical_dtype": str(series.dtype),
        "semantic_type": semantic.get("semantic_type"),
        "value_function": semantic.get("value_function"),
        "numeric_meaning": semantic.get("numeric_meaning"),
        "semantic_confidence": semantic.get("confidence"),
        "concept_description": semantic.get("concept_description"),
        "identifier_or_entity_reference_protected": protected,
        "affirmative_preservation_evidence": (
            preservation_evidence if affirmative_preservation else None
        ),
        "supplied_consequence_evidence": supplied_consequence,
        "median_semantic_eligibility": {
            "status": semantic_result["semantic_eligible"],
            "reason_code": semantic_reason,
        },
        "median_mechanical_preflight": mechanical,
    }
    if semantic_reason == "high_cardinality_integral_label_risk":
        semantic_evidence = semantic_result["evidence"]
        readiness["numeric_label_risk_evidence"] = {
            key: semantic_evidence[key]
            for key in ("integral_fraction", "unique_ratio", "repetition_rate")
        }
    return readiness


def _missing_values_prompt(
    issues: Sequence[DetectedIssue],
    profile: SemanticProfile,
    dataframe: pd.DataFrame | None,
) -> str:
    payload = [{
        "issue_id": issue.issue_id,
        "issue_type": issue.issue_type,
        "target_columns": issue.target_columns,
        "readiness": _missing_value_readiness(issue, profile, dataframe),
    } for issue in issues]
    context = profile.dataset_context
    return f"""
Return exactly one compact decision for each missing_values issue. Copy its exact
issue_id; do not omit, duplicate, or invent IDs. Keep rationale to one short
sentence grounded only in the supplied evidence.

MISSING-VALUE DECISION RUBRIC

FIRST SEPARATE ACTION AVAILABILITY FROM THE DECISION TYPE:
Protection or failed eligibility means that automatic imputation is unavailable.
It is not affirmative evidence that the missing state is legitimate and must not
be used as preservation evidence for no_action. This includes identifier/entity-
reference protection, temporal semantics, and failed semantic or mechanical
preflight. If action is unavailable, affirmative_preservation_evidence is null,
and no supplied consequence supports review, use insufficient_evidence.

ACTION:
Use only when filling is semantically and contextually appropriate and a
supported strategy is feasible. For the currently supported automatic numeric
path return decision_type="action", proposed_action="impute_missing", and
parameters={{"strategy":"median"}}, with
recommendation_basis="semantically_appropriate_imputation". This is a
RECOMMENDATION ONLY. You do not
authorize execution; deterministic semantic and mechanical policy independently
revalidates it.

NO_ACTION:
Use only when affirmative_preservation_evidence concretely establishes that the
missing state should be preserved. Cite that concrete evidence in the rationale.
Return recommendation_basis="affirmative_preservation_evidence" and a non-empty
preservation_evidence_keys list naming the supplied evidence used.
Valid preservation evidence is an explicit supplied fact that absence genuinely
means not applicable or otherwise has an approved meaningful interpretation.
Only keys inside a non-null affirmative_preservation_evidence object may be
listed. Protection, temporal type, failed readiness, inability to impute, low
missingness, or the absence of a reason to fill are not preservation evidence.
Low missingness alone is not a preservation reason. Uncertainty is not a reason
for no_action.

INSUFFICIENT_EVIDENCE:
Use when evidence is inadequate or contradictory and it is not possible to
determine whether filling is semantically appropriate. Return proposed_action="no_op".
Return recommendation_basis="insufficient_or_contradictory_evidence".
A false or unknown readiness result does not by itself prove that missingness is
legitimate.

FLAG_FOR_REVIEW:
Use when evidence indicates the missingness is consequential or problematic but
automatic filling is inappropriate or unsafe. Return proposed_action="flag_rows".
Select the review_basis supported by the supplied readiness evidence.
Do not map every failed readiness check to review; uncertainty without evidence
of consequence is insufficient_evidence.

Never fabricate identifiers/entity references or temporal values. Never propose
median imputation for temporal, identifier, entity-reference, ordinal, nominal
code, non-numeric, semantically false, or semantically unknown cases. Advisory
preflight facts do not override the final deterministic policy.

Shared dataset context: row_entity={context.row_entity!r}, probable_domain={context.probable_domain!r}

Missing-value issues and compact readiness evidence:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)}

Return structured data only.
""".strip()


def _is_affirmative(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return False


def _validate_missing_value_alignment(
    decision: Tier2MissingDecisionPayload,
    readiness: Dict[str, Any],
) -> None:
    """Reject contradictions without selecting a replacement decision."""
    semantic = readiness["median_semantic_eligibility"]
    mechanical = readiness["median_mechanical_preflight"]
    preservation = readiness.get("affirmative_preservation_evidence")

    if decision.decision_type == "action":
        failures = []
        if semantic.get("status") != "true":
            failures.append("median semantic eligibility is not true")
        if mechanical.get("status") != "pass":
            failures.append("median mechanical preflight did not pass")
        if readiness.get("identifier_or_entity_reference_protected") is not False:
            failures.append("identifier/entity-reference protection is not false")
        if readiness.get("numeric_meaning") != "magnitude":
            failures.append("numeric_meaning is not magnitude")
        if readiness.get("value_function") != "measurement":
            failures.append("value_function is not measurement")
        if readiness.get("semantic_type") not in QUANTITATIVE_TYPES:
            failures.append("semantic_type is not an approved quantitative type")
        if preservation:
            failures.append("affirmative preservation evidence exists")
        if failures:
            raise ValueError("Invalid missing-value median recommendation: " + "; ".join(failures))
        return

    if decision.decision_type == "no_action":
        if not isinstance(preservation, dict) or not preservation:
            raise ValueError("Missing-value no_action requires supplied affirmative preservation evidence.")
        invalid_keys = [
            key for key in decision.preservation_evidence_keys
            if key not in preservation or not _is_affirmative(preservation[key])
        ]
        if invalid_keys:
            raise ValueError(
                "Missing-value no_action references absent or non-affirmative evidence: "
                + ", ".join(invalid_keys)
            )
        return

    if decision.decision_type == "flag_for_review":
        basis = decision.review_basis
        valid = {
            "protected_entity_reference": (
                readiness.get("identifier_or_entity_reference_protected") is True
            ),
            "temporal_value_not_safely_synthesizable": (
                readiness.get("value_function") == "temporal"
                or readiness.get("semantic_type") in {"date", "time", "datetime"}
            ),
            "high_missingness_requires_review": (
                readiness.get("missingness_band") in {"high", "very_high"}
            ),
            "other_supplied_consequence": _is_affirmative(
                readiness.get("supplied_consequence_evidence")
            ),
        }[basis]
        if not valid:
            raise ValueError(f"Missing-value review_basis is unsupported by readiness: {basis}")


def _normalize_missing_value_decision(
    decision: Tier2MissingDecisionPayload,
) -> Tier2DecisionPayload:
    return Tier2DecisionPayload.model_validate({
        "issue_id": decision.issue_id,
        "decision_type": decision.decision_type,
        "proposed_action": decision.proposed_action,
        "parameters": decision.parameters.model_dump(),
        "confidence": decision.confidence,
        "rationale": decision.rationale,
    })


def _validate_coverage(
    expected: Sequence[str], decisions: Sequence[Tier2DecisionPayload]
) -> Dict[str, Tier2DecisionPayload]:
    actual = [decision.issue_id for decision in decisions]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("Tier 2 decision output does not exactly cover input issue IDs.")
    return {decision.issue_id: decision for decision in decisions}


def _fallback(issue: DetectedIssue, semantic: Dict[str, Any], reason: str) -> CleaningDecision:
    return CleaningDecision(
        decision_id=_decision_id(issue.issue_id), source_issue_ids=[issue.issue_id],
        target_columns=issue.target_columns, decision_type="insufficient_evidence",
        proposed_action="no_op", confidence=0.0,
        rationale="Tier 2 decision inference failed; data is preserved.",
        evidence_summary={"fallback_reason": reason, **issue.evidence},
        semantic_basis=semantic, risk_level="high", requires_review=True,
        decision_source="fallback",
    )


def _apply_llm_policy(
    decision: Tier2DecisionPayload, issue: DetectedIssue, profile: SemanticProfile
) -> CleaningDecision:
    semantic = _semantic_for(issue, profile)
    semantic_confidences = [value.get("confidence", 0.0) for value in semantic.values()]
    semantic_cap = min(semantic_confidences) if semantic_confidences else 0.5
    risky = decision.proposed_action in {
        "normalize_categorical_labels", "parse_dates", "coerce_dtype",
        "impute_missing", "add_missing_indicator", "drop_duplicate_rows",
    }
    requires_review = risky or decision.decision_type in {
        "flag_for_review", "insufficient_evidence",
    }
    risk_level = "high" if requires_review else "low"
    return CleaningDecision.model_validate({
        "decision_id": _decision_id(issue.issue_id),
        "source_issue_ids": [issue.issue_id],
        "target_columns": issue.target_columns,
        "decision_type": decision.decision_type,
        "proposed_action": decision.proposed_action,
        "parameters": decision.parameters,
        "confidence": min(decision.confidence, semantic_cap, 0.9 if risky else 1.0),
        "rationale": decision.rationale,
        "evidence_summary": issue.evidence,
        "semantic_basis": semantic,
        "risk_level": risk_level,
        "requires_review": requires_review,
        "decision_source": "llm_tier2",
    })


def _salvage_tier2_response(
    batch_kind: str,
    issues: Sequence[DetectedIssue],
    parsed_decisions: Sequence[Any],
    semantic_profile: SemanticProfile,
    dataframe: pd.DataFrame | None,
) -> Dict[str, Any]:
    """Accept uniquely matched valid decisions and isolate every unresolved ID."""
    issue_index = {issue.issue_id: issue for issue in issues}
    expected_ids = list(issue_index)
    decisions_by_id: Dict[str, List[Any]] = {}
    unexpected_ids = []
    for decision in parsed_decisions:
        issue_id = decision.issue_id
        if issue_id not in issue_index:
            unexpected_ids.append(issue_id)
            continue
        decisions_by_id.setdefault(issue_id, []).append(decision)

    missing_ids = [issue_id for issue_id in expected_ids if issue_id not in decisions_by_id]
    duplicate_ids = [
        issue_id for issue_id in expected_ids
        if len(decisions_by_id.get(issue_id, ())) > 1
    ]
    rejected = []
    accepted: Dict[str, CleaningDecision] = {}
    readiness_index = (
        {
            issue.issue_id: _missing_value_readiness(
                issue, semantic_profile, dataframe,
            )
            for issue in issues
        }
        if batch_kind == "missing_values" else {}
    )

    for issue_id in expected_ids:
        matches = decisions_by_id.get(issue_id, ())
        if not matches:
            rejected.append({"issue_id": issue_id, "reason": "missing_issue_id"})
            continue
        if len(matches) > 1:
            rejected.append({"issue_id": issue_id, "reason": "duplicate_issue_id"})
            continue
        try:
            decision = matches[0]
            if batch_kind == "missing_values":
                _validate_missing_value_alignment(decision, readiness_index[issue_id])
                decision = _normalize_missing_value_decision(decision)
            accepted[issue_id] = _apply_llm_policy(
                decision, issue_index[issue_id], semantic_profile,
            )
        except Exception as exc:
            rejected.append({
                "issue_id": issue_id,
                "reason": f"{type(exc).__name__}: {exc}",
            })

    return {
        "accepted": accepted,
        "unresolved_issue_ids": [
            issue_id for issue_id in expected_ids if issue_id not in accepted
        ],
        "rejected": rejected,
        "coverage": {
            "missing_issue_ids": missing_ids,
            "duplicate_issue_ids": duplicate_ids,
            "unexpected_issue_ids": unexpected_ids,
        },
    }


def decide_cleaning(
    issues: Sequence[DetectedIssue], semantic_profile: SemanticProfile,
    llm_provider: LLMProvider, config: DecisionConfig = DecisionConfig(),
    dataframe: pd.DataFrame | None = None,
) -> tuple[List[CleaningDecision], Dict[str, Any], List[str]]:
    started = time.perf_counter()
    deterministic_started = time.perf_counter()
    decisions, tier2 = [], []
    for issue in issues:
        semantic = _semantic_for(issue, semantic_profile)
        resolved = _tier1(issue, semantic)
        (decisions if resolved else tier2).append(resolved or issue)
    deterministic_seconds = time.perf_counter() - deterministic_started
    warnings, llm_seconds, calls, retries, failures = [], 0.0, 0, 0, 0
    successful_batches = 0
    llm_call_metrics = []
    batch_sizes = []
    general_tier2 = [issue for issue in tier2 if issue.issue_type != "missing_values"]
    missing_tier2 = [issue for issue in tier2 if issue.issue_type == "missing_values"]
    batches = [
        ("general", general_tier2[start:start + config.max_issues_per_batch])
        for start in range(0, len(general_tier2), config.max_issues_per_batch)
    ] + [
        ("missing_values", missing_tier2[start:start + config.max_issues_per_batch])
        for start in range(0, len(missing_tier2), config.max_issues_per_batch)
    ]
    batch_kinds = []
    recovery_audit = []
    fallback_issue_count = 0
    for original_batch_index, (batch_kind, batch) in enumerate(batches, start=1):
        batch_sizes.append(len(batch))
        batch_kinds.append(batch_kind)
        original_issue_ids = [issue.issue_id for issue in batch]
        issue_index = {issue.issue_id: issue for issue in batch}
        accepted: Dict[str, CleaningDecision] = {}
        last_rejection = {}
        batch_audit = {
            "original_batch_index": original_batch_index,
            "batch_kind": batch_kind,
            "original_issue_ids": original_issue_ids,
            "accepted_on_first_response": [],
            "recovered_on_retry": [],
            "final_fallback_issue_ids": [],
            "calls": [],
        }

        def call_subset(
            subset: Sequence[DetectedIssue], stage: str,
        ) -> Dict[str, Any]:
            nonlocal calls, retries, llm_seconds
            if stage != "initial":
                retries += 1
            calls += 1
            call_started = time.perf_counter()
            prompt = (
                _missing_values_prompt(subset, semantic_profile, dataframe)
                if batch_kind == "missing_values"
                else _prompt(subset, semantic_profile)
            )
            subset_ids = [issue.issue_id for issue in subset]
            metric = {
                "batch_index": original_batch_index,
                "attempt": len(batch_audit["calls"]) + 1,
                "recovery_stage": stage,
                "batch_kind": batch_kind,
                "issue_ids": subset_ids,
                "retry_batch_size": len(subset),
                "prompt_character_length": len(prompt),
            }
            call_audit = {
                "stage": stage,
                "issue_ids": subset_ids,
                "retry_batch_size": len(subset),
            }
            try:
                response_model = (
                    Tier2MissingDecisionBatch
                    if batch_kind == "missing_values"
                    else Tier2DecisionBatch
                )
                raw = llm_provider.generate_structured(prompt, response_model)
                elapsed = time.perf_counter() - call_started
                llm_seconds += elapsed
                native = getattr(llm_provider, "last_response_metadata", None) or {}
                parsed = response_model(**raw)
                result = _salvage_tier2_response(
                    batch_kind, subset, parsed.decisions,
                    semantic_profile, dataframe,
                )
                metric.update({
                    "api_wall_duration_seconds": round(elapsed, 6),
                    **{key: native.get(key) for key in (
                        "prompt_eval_count", "prompt_eval_duration", "eval_count",
                        "eval_duration", "total_duration", "load_duration", "done_reason",
                    )},
                })
                call_audit.update({
                    "provider_status": "success",
                    "accepted_issue_ids": list(result["accepted"]),
                    "rejected": result["rejected"],
                    "unresolved_issue_ids": result["unresolved_issue_ids"],
                    "coverage": result["coverage"],
                })
                result["batch_wide_failure"] = False
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - call_started
                llm_seconds += elapsed
                native = getattr(llm_provider, "last_response_metadata", None) or {}
                error = f"{type(exc).__name__}: {exc}"
                is_timeout = "timeout" in error.lower() or "timed out" in error.lower()
                metric.update({
                    "api_wall_duration_seconds": round(elapsed, 6),
                    "error": error,
                    **{key: native.get(key) for key in (
                        "prompt_eval_count", "prompt_eval_duration", "eval_count",
                        "eval_duration", "total_duration", "load_duration", "done_reason",
                    )},
                })
                call_audit.update({
                    "provider_status": "timeout" if is_timeout else "error",
                    "accepted_issue_ids": [],
                    "rejected": [
                        {"issue_id": issue_id, "reason": error}
                        for issue_id in subset_ids
                    ],
                    "unresolved_issue_ids": subset_ids,
                })
                return {
                    "accepted": {},
                    "unresolved_issue_ids": subset_ids,
                    "rejected": call_audit["rejected"],
                    "coverage": None,
                    "batch_wide_failure": True,
                }
            finally:
                llm_call_metrics.append(metric)
                batch_audit["calls"].append(call_audit)

        def retain(result: Dict[str, Any], first_response: bool) -> List[str]:
            accepted.update(result["accepted"])
            accepted_ids = list(result["accepted"])
            destination = (
                "accepted_on_first_response" if first_response
                else "recovered_on_retry"
            )
            batch_audit[destination].extend(accepted_ids)
            for item in result["rejected"]:
                last_rejection[item["issue_id"]] = item["reason"]
            return result["unresolved_issue_ids"]

        initial_result = call_subset(batch, "initial")
        unresolved_ids = retain(initial_result, True)

        if unresolved_ids and config.retry_count > 0:
            unresolved = [issue_index[issue_id] for issue_id in unresolved_ids]
            if initial_result["batch_wide_failure"] and len(unresolved) == 2:
                retry_groups = [[issue] for issue in unresolved]
            else:
                retry_groups = [
                    unresolved[start:start + 2]
                    for start in range(0, len(unresolved), 2)
                ]

            remaining_ids = []
            for retry_group in retry_groups:
                retry_result = call_subset(retry_group, "retry")
                group_unresolved = retain(retry_result, False)
                coverage = retry_result.get("coverage") or {}
                coverage_ambiguous = bool(
                    coverage.get("missing_issue_ids")
                    or coverage.get("duplicate_issue_ids")
                )
                if (
                    group_unresolved and len(retry_group) > 1
                    and (retry_result["batch_wide_failure"] or coverage_ambiguous)
                ):
                    for issue_id in group_unresolved:
                        singleton_result = call_subset(
                            [issue_index[issue_id]], "singleton_retry",
                        )
                        remaining_ids.extend(retain(singleton_result, False))
                else:
                    remaining_ids.extend(group_unresolved)
            unresolved_ids = list(dict.fromkeys(remaining_ids))

        if unresolved_ids:
            failures += 1
            fallback_issue_count += len(unresolved_ids)
            batch_audit["final_fallback_issue_ids"] = unresolved_ids
            for issue_id in unresolved_ids:
                issue = issue_index[issue_id]
                reason = last_rejection.get(issue_id, "Tier 2 issue remained unresolved.")
                warnings.append(
                    f"Tier 2 decision failed for issue {issue_id}: {reason}"
                )
                accepted[issue_id] = _fallback(
                    issue, _semantic_for(issue, semantic_profile), reason,
                )
        else:
            successful_batches += 1
        decisions.extend(accepted[issue.issue_id] for issue in batch)
        recovery_audit.append(batch_audit)
    order = {issue.issue_id: index for index, issue in enumerate(issues)}
    decisions.sort(key=lambda decision: order[decision.source_issue_ids[0]])
    summary = {
        "total_decisions": len(decisions),
        "by_decision_type": dict(Counter(d.decision_type for d in decisions)),
        "by_proposed_action": dict(Counter(d.proposed_action for d in decisions)),
        "tier1_decisions": sum(d.decision_source == "deterministic_tier1" for d in decisions),
        "tier2_decisions": sum(d.decision_source == "llm_tier2" for d in decisions),
        "fallback_decisions": sum(d.decision_source == "fallback" for d in decisions),
        "deterministic_runtime_seconds": round(deterministic_seconds, 6),
        "tier2_issue_count": len(tier2), "tier2_batch_sizes": batch_sizes,
        "tier2_batch_kinds": batch_kinds,
        "llm_decision_calls": calls, "llm_runtime_seconds": round(llm_seconds, 6),
        "intended_llm_decision_calls": len(batch_sizes),
        "retries": retries, "successful_batches": successful_batches,
        "batch_failures": failures,
        "fallback_issue_count": fallback_issue_count,
        "tier2_recovery_audit": recovery_audit,
        "llm_calls": llm_call_metrics,
        "total_runtime_seconds": round(time.perf_counter() - started, 6),
    }
    return decisions, summary, warnings
