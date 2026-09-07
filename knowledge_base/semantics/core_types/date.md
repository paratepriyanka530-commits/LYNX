---
knowledge_id: SEM-006
category: semantic
topic: date
version: 1.0
status: active
---

# Date

## Definition
A calendar date where day/month/year information is meaningful and time-of-day is absent or semantically irrelevant.

## Semantic Indicators
Calendar-formatted values and names such as date, birth_date, join_date, due_date.

## Important Distinctions
Distinguish date from datetime. A midnight timestamp does not imply meaningful time-of-day.

## Quality Expectations
Valid calendar values, consistent interpretation of ambiguous formats, plausible ordering.

## Possible Data Quality Problems
Mixed formats, invalid dates, day/month ambiguity, placeholder dates, parse failures.

## Diagnostic Evidence
Check distributions, frequencies, samples, pattern consistency, domain context, related columns, repeated occurrences, units, and whether an apparent anomaly is reproducible.

## Cross-Column Relationships
Start/end and birth/event relationships may validate dates.

## Cleaning Decision Guidance
Standardize representation after interpretation is sufficiently certain; do not silently guess ambiguous day/month order.

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
