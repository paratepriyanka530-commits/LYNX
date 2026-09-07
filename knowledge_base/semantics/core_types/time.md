---
knowledge_id: SEM-007
category: semantic
topic: time
version: 1.0
status: active
---

# Time

## Definition
A clock time or time-of-day without an independently meaningful calendar date.

## Semantic Indicators
HH:MM, HH:MM:SS, AM/PM forms, names such as time, opening_time.

## Important Distinctions
Distinguish time from datetime and durations. A value containing a colon is not sufficient proof.

## Quality Expectations
Clock components should be valid and consistent with intended timezone/format where relevant.

## Possible Data Quality Problems
Invalid hours/minutes, mixed 12/24-hour formats, timezone ambiguity.

## Diagnostic Evidence
Check distributions, frequencies, samples, pattern consistency, domain context, related columns, repeated occurrences, units, and whether an apparent anomaly is reproducible.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Normalize only after establishing time semantics and timezone assumptions.

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
