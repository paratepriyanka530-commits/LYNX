---
knowledge_id: SEM-008
category: semantic
topic: datetime
version: 1.0
status: active
---

# Datetime

## Definition
A temporal value where both calendar date and time-of-day carry semantic information.

## Semantic Indicators
Combined date-time values, timestamps, event times, created_at, sensor timestamps.

## Important Distinctions
Distinguish true datetime from date values serialized at midnight and from standalone time.

## Quality Expectations
Calendar and clock components should be valid; timezone handling should be coherent when present.

## Possible Data Quality Problems
Mixed timezone forms, malformed timestamps, inconsistent precision, date-only values mixed into timestamps.

## Diagnostic Evidence
Check distributions, frequencies, samples, pattern consistency, domain context, related columns, repeated occurrences, units, and whether an apparent anomaly is reproducible.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Do not discard timezone or time-of-day information without evidence that it is irrelevant.

## When NOT to Clean
Do not modify unusual but plausible values, meaningful missingness, rare categories, valid alternate representations, or values whose intended correction cannot be supported.

## Evidence Required Before Modification
Destructive or value-changing actions require strong evidence. Weak evidence should lead to no_action, flag_for_review, or needs_additional_context.

## Reasoning Principles
- Physical dtype is not semantic meaning.
- Statistical outlier does not automatically mean error.
- Missing does not automatically mean impute.
- Never invent a replacement value.
- Prefer deterministic validation when possible.
- Preserve uncertainty and information.
- no_action is valid.
