"""Stable, privacy-aware column manifest for the Profiling workspace."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from api.dataset_overview import serialize_json_cell
from pipelines.pipeline1_cleaning.techniques.privacy_masking import REDACTION_VALUE


UNRESOLVED_SEMANTIC_VALUES = {"", "unknown", "unresolved", "not_available"}
ATTENTION_STATUSES = {"proposed", "skipped", "requires_review", "failed"}
STATUS_PRIORITY = {
    "failed": 0,
    "requires_review": 1,
    "proposed": 2,
    "skipped": 3,
    "applied": 4,
    "not_required": 5,
    "unavailable": 6,
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _records(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _percentage(value: Any) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _action_status(
    decision: Mapping[str, Any],
    instruction: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> str:
    decision_type = str(decision.get("decision_type") or "")
    if bool(decision.get("requires_review")) or decision_type == "flag_for_review":
        return "requires_review"
    if decision_type == "insufficient_evidence":
        return "unavailable"
    if decision_type == "no_action":
        return "not_required"
    if decision_type != "action":
        return "unavailable"
    if execution.get("status") == "failed":
        return "failed"
    if execution.get("status") == "success":
        return "applied"
    if instruction and str(instruction.get("action_id") or "") == "no_op":
        return "skipped"
    return "proposed"


def _action_record(
    decision: Mapping[str, Any],
    instruction: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    execution_status = execution.get("status")
    return {
        "decision_id": str(decision.get("decision_id") or "unavailable"),
        "action_id": str(decision.get("proposed_action") or "no_op"),
        "decision_type": str(decision.get("decision_type") or "insufficient_evidence"),
        "status": _action_status(decision, instruction, execution),
        "rationale": _optional_text(decision.get("rationale")),
        "evidence": dict(_mapping(decision.get("evidence_summary"))),
        "requires_review": bool(decision.get("requires_review")),
        "execution_status": (
            str(execution_status) if execution_status in {"success", "failed"} else None
        ),
        "changed": bool(execution.get("changed")) if execution else None,
        "changed_cell_count": (
            _non_negative_int(execution.get("changed_cell_count"))
            if execution
            else None
        ),
    }


def _profile_status(
    report: Mapping[str, Any],
    profiles: list[Mapping[str, Any]],
    semantics_by_name: Mapping[str, Mapping[str, Any]],
) -> str:
    if str(report.get("status") or "").casefold() == "failed":
        return "failed"
    if not profiles:
        return "unavailable"
    profiled_names = {str(item.get("column_name")) for item in profiles}
    resolved_names = profiled_names.intersection(semantics_by_name)
    telemetry = _mapping(_mapping(report.get("semantic_profile")).get("telemetry"))
    if len(resolved_names) < len(profiled_names) or _non_negative_int(telemetry.get("batch_failures")):
        return "partial"
    return "completed"


def build_profiling_manifest(
    report: Mapping[str, Any],
    *,
    dataset_name: str,
    profiled_at: str | None,
    privacy_clearance_passed: bool,
    pii_columns: Iterable[str] = (),
) -> dict[str, Any]:
    """Join observed, semantic, decision, and execution evidence by column."""
    before = _mapping(report.get("before_profile"))
    profiles = _records(before.get("column_profiles"))
    semantic_profile = _mapping(report.get("semantic_profile"))
    semantics = _records(semantic_profile.get("columns"))
    semantics_by_name = {
        str(item.get("column_name")): item
        for item in semantics
        if item.get("column_name") is not None
    }
    decisions = _records(report.get("decisions"))
    instructions_by_decision = {
        str(item.get("source_decision_id")): item
        for item in _records(report.get("instructions"))
        if item.get("source_decision_id") is not None
    }
    executions_by_decision = {
        str(item.get("source_decision_id")): item
        for item in _records(report.get("executions"))
        if item.get("source_decision_id") is not None
    }
    protected = {str(column) for column in pii_columns}

    columns: list[dict[str, Any]] = []
    for column_index, profile in enumerate(profiles):
        column_name = str(profile.get("column_name") or "")
        semantic = semantics_by_name.get(column_name, {})
        missing = _mapping(profile.get("missing"))
        uniqueness = _mapping(profile.get("uniqueness"))
        column_decisions = [
            decision
            for decision in decisions
            if column_name in [str(value) for value in decision.get("target_columns", [])]
        ]
        actions = []
        for decision in column_decisions:
            decision_id = str(decision.get("decision_id") or "")
            actions.append(
                _action_record(
                    decision,
                    instructions_by_decision.get(decision_id, {}),
                    executions_by_decision.get(decision_id, {}),
                )
            )
        actions.sort(key=lambda action: STATUS_PRIORITY[action["status"]])
        primary_action = actions[0] if actions else None
        column_status = primary_action["status"] if primary_action else "not_required"

        privacy_restricted = not privacy_clearance_passed or column_name in protected
        raw_examples = profile.get("sample_values")
        observed_examples: list[Any] = []
        if privacy_clearance_passed and isinstance(raw_examples, list):
            if column_name in protected:
                if any(serialize_json_cell(value) is not None for value in raw_examples):
                    observed_examples = [REDACTION_VALUE]
            else:
                observed_examples = [serialize_json_cell(value) for value in raw_examples[:5]]

        explanation = None
        explanation_source = "unavailable"
        if primary_action and primary_action.get("rationale"):
            explanation = primary_action["rationale"]
            explanation_source = "cleaning_decision"
        elif _optional_text(semantic.get("reasoning_summary")):
            explanation = _optional_text(semantic.get("reasoning_summary"))
            explanation_source = "semantic_reasoning"
        elif _optional_text(semantic.get("concept_description")):
            explanation = _optional_text(semantic.get("concept_description"))
            explanation_source = "semantic_description"

        columns.append({
            "column_index": column_index,
            "column_name": column_name,
            "physical_dtype": _optional_text(profile.get("physical_dtype")),
            "semantic_role": _optional_text(semantic.get("semantic_role")),
            "semantic_type": _optional_text(semantic.get("semantic_type")),
            "meaning": _optional_text(semantic.get("concept_description")),
            "semantic_confidence": semantic.get("confidence") if isinstance(semantic.get("confidence"), (int, float)) else None,
            "non_null_count": _non_negative_int(profile.get("non_null_count")),
            "missing_count": _non_negative_int(missing.get("count")),
            "missing_percentage": _percentage(missing.get("percentage")),
            "unique_count": _non_negative_int(uniqueness.get("unique_count")),
            "primary_action": primary_action,
            "actions": actions,
            "status": column_status,
            "decision_explanation": explanation,
            "explanation_source": explanation_source,
            "observed_examples": observed_examples,
            "privacy_restricted": privacy_restricted,
        })

    resolved_roles = sorted({
        str(column["semantic_role"])
        for column in columns
        if column.get("semantic_role")
        and str(column["semantic_role"]).strip().casefold() not in UNRESOLVED_SEMANTIC_VALUES
    }, key=str.casefold)
    attention_columns = sum(column["status"] in ATTENTION_STATUSES for column in columns)
    status = _profile_status(report, profiles, semantics_by_name)
    return {
        "dataset_name": dataset_name,
        "profiled_at": profiled_at,
        "profile_status": status,
        "total_columns": len(columns),
        "semantic_roles_detected": resolved_roles,
        "semantic_role_count": len(resolved_roles),
        "columns_requiring_cleaning": attention_columns,
        "columns": columns,
    }


__all__ = ["build_profiling_manifest"]
