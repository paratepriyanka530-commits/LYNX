"""Build citable evidence artifacts from saved missing-value diagnostics.

This utility is diagnostic-only. It reads recorded responses and validation
results; it does not call an LLM or import/modify Pipeline 1 behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DIAGNOSTICS_DIR = Path(__file__).resolve().parent / "diagnostics"
JSON_OUTPUT = DIAGNOSTICS_DIR / "missing_value_validator_rejections.json"
MARKDOWN_OUTPUT = DIAGNOSTICS_DIR / "missing_value_validator_rejections.md"


def _decision_for_issue(raw_response: str, issue_id: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError):
        return None

    for decision in payload.get("decisions", []):
        if decision.get("issue_id") == issue_id:
            return decision
    return None


def _llm_claim(decision: dict[str, Any], readiness: dict[str, Any]) -> str:
    rationale = decision.get("rationale") or "No rationale was supplied."
    keys = decision.get("preservation_evidence_keys") or []
    cited = ", ".join(str(key) for key in keys) or "none"
    preservation = readiness.get("affirmative_preservation_evidence")
    return (
        f'{rationale} Cited preservation_evidence_keys=[{cited}]; '
        f"supplied affirmative_preservation_evidence={preservation!r}."
    )


def _final_outcome(case: dict[str, Any]) -> str:
    final = case.get("final_decision") or {}
    decision_type = final.get("decision_type", "unknown")
    action = final.get("proposed_action", "unknown")
    source = final.get("decision_source", "unknown")
    return f"{decision_type} / {action} ({source})"


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def build() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []

    for path in sorted(DIAGNOSTICS_DIR.glob("*.json")):
        if path == JSON_OUTPUT:
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        records = document if isinstance(document, list) else []
        response_case_count = 0
        rejection_count = 0

        for record in records:
            for case in record.get("cases", []):
                response_case_count += 1
                final = case.get("final_decision") or {}
                issue_ids = final.get("source_issue_ids") or []
                if not issue_ids:
                    continue
                issue_id = issue_ids[0]
                attempts = [
                    (
                        "first",
                        case.get("first_raw_qwen_response"),
                        case.get("first_validation"),
                    )
                ]
                attempts.extend(
                    (
                        attempt.get("stage", "retry"),
                        attempt.get("raw_qwen_response"),
                        attempt.get("validation"),
                    )
                    for attempt in case.get("retry_attempts", [])
                )

                for attempt_name, raw_response, validation in attempts:
                    if not raw_response or not validation or validation == "accepted":
                        continue
                    decision = _decision_for_issue(raw_response, issue_id)
                    if decision is None:
                        continue
                    rows.append(
                        {
                            "column_name": case.get("column", "unknown"),
                            "llm_claim": _llm_claim(decision, case.get("readiness") or {}),
                            "validator_rejection_rule": validation,
                            "safe_final_outcome": _final_outcome(case),
                            "diagnostic_file": path.name,
                            "attempt": attempt_name,
                            "issue_id": issue_id,
                        }
                    )
                    rejection_count += 1

        scanned.append(
            {
                "diagnostic_file": path.name,
                "response_case_count": response_case_count,
                "validator_rejection_rows": rejection_count,
            }
        )

    artifact = {
        "scope": "Recorded missing-value Tier-2 responses rejected by readiness alignment",
        "source_files": scanned,
        "rejection_count": len(rows),
        "rejections": rows,
    }
    JSON_OUTPUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    lines = [
        "# Missing-value Tier-2 validator rejections",
        "",
        "| Column name | What the LLM claimed / fabricated | Validator field or rule | Safe final outcome |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_escape(row[key])
                for key in (
                    "column_name",
                    "llm_claim",
                    "validator_rejection_rule",
                    "safe_final_outcome",
                )
            )
            + " |"
        )
    MARKDOWN_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":
    result = build()
    print(f"Wrote {result['rejection_count']} rejection rows to:")
    print(JSON_OUTPUT)
    print(MARKDOWN_OUTPUT)
