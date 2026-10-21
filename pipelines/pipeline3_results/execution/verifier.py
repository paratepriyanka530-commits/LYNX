"""Deterministic verification of computed analytical results."""

from __future__ import annotations

import math

from pipelines.pipeline3_results.execution.schemas import (
    ResultVerification,
    VerifiedResult,
)
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan


def _finite(value: object) -> bool:
    return not isinstance(value, float) or math.isfinite(value)


def verify_result(result: VerifiedResult, plan: AnalyticalPlan) -> ResultVerification:
    checks: list[str] = []
    warnings: list[str] = list(result.warnings)
    errors: list[str] = []

    if result.plan_id != plan.plan_id:
        errors.append("Result plan_id does not match the executed plan.")
    else:
        checks.append("plan_id matched")

    coverage = result.coverage
    if not (coverage.usable_rows <= coverage.filtered_rows <= coverage.source_rows):
        errors.append("Coverage row counts do not reconcile.")
    else:
        checks.append("coverage row counts reconciled")

    field_names = [field.name for field in result.schema_.fields]
    if result.rows and any(set(row) != set(field_names) for row in result.rows):
        errors.append("Result rows do not match the declared schema.")
    else:
        checks.append("result rows matched schema")

    if result.scalar is not None and not _finite(result.scalar):
        errors.append("Scalar result is non-finite.")
    if any(not _finite(value) for row in result.rows for value in row.values()):
        errors.append("Result rows contain non-finite numeric values.")
    if not errors:
        checks.append("numeric outputs were finite")

    if result.status == "READY" and result.scalar is None and not result.rows:
        errors.append("READY result is unexpectedly empty.")

    if plan.operation == "TREND" and result.rows:
        temporal = plan.temporal[0].column
        values = [str(row[temporal]) for row in result.rows]
        if values != sorted(values):
            errors.append("Temporal result is not chronologically sorted.")
        else:
            checks.append("temporal result was sorted")

    if plan.operation in {"RANK", "TOP_N", "BOTTOM_N"} and result.rows:
        values = [float(row["value"]) for row in result.rows]
        descending = plan.sort[0].direction == "DESC"
        if values != sorted(values, reverse=descending):
            errors.append("Rank result is not ordered as requested.")
        else:
            checks.append("rank ordering verified")

    if plan.operation == "PART_TO_WHOLE" and result.rows:
        total_share = sum(float(row["share_of_total"]) for row in result.rows)
        if not math.isclose(total_share, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            errors.append("Part-to-whole shares do not reconcile to 1.0.")
        else:
            checks.append("part-to-whole shares reconciled")

    grouped_total = result.execution_metadata.details.get("grouped_total")
    aggregate_total = result.execution_metadata.details.get("aggregate_total")
    if grouped_total is not None and aggregate_total is not None:
        if not math.isclose(float(grouped_total), float(aggregate_total), rel_tol=1e-9, abs_tol=1e-9):
            errors.append("Grouped values do not reconcile to the aggregate total.")
        else:
            checks.append("grouped totals reconciled")

    if errors:
        status = "FAILED"
    elif warnings:
        status = "VERIFIED_WITH_WARNINGS"
    else:
        status = "VERIFIED"
    return ResultVerification(status=status, checks=checks, warnings=warnings, errors=errors)


__all__ = ["verify_result"]
