"""Structured deterministic issue records and quality summaries."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class DetectedIssue(BaseModel):
    issue_id: str = Field(min_length=1)
    issue_type: str = Field(min_length=1)
    target_columns: List[str] = Field(default_factory=list)
    affected_count: int = Field(ge=0)
    affected_percentage: float = Field(ge=0.0, le=100.0)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    examples: List[Any] = Field(default_factory=list)
    status: Literal["issue", "candidate", "observation"]
    risk_hint: Literal["low", "medium", "high", "context_dependent"]
    detector: str = "deterministic_issue_detector"


def summarize_issues(issues: List[DetectedIssue]) -> Dict[str, Any]:
    by_type = Counter(issue.issue_type for issue in issues)
    by_status = Counter(issue.status for issue in issues)
    columns = sorted({column for issue in issues for column in issue.target_columns})
    dataset_level = sum(1 for issue in issues if not issue.target_columns)
    return {
        "total_issues_detected": len(issues),
        "issues_by_type": dict(sorted(by_type.items())),
        "issues_by_status": dict(sorted(by_status.items())),
        "affected_columns": columns,
        "dataset_level_issue_count": dataset_level,
    }
