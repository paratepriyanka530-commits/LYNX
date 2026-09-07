---
knowledge_id: SEM-005
category: semantic
topic: boolean
version: 1.0
status: active
---

# Boolean

## Definition
A two-state logical concept such as true/false, yes/no, active/inactive, present/absent.

## Semantic Indicators
Two dominant semantic states, including textual or numeric encodings.

## Important Distinctions
Not every two-valued column is boolean; binary categories such as sex, treatment group, or region code may be categorical.

## Quality Expectations
Equivalent truth-state encodings should be consistent.

## Possible Data Quality Problems
Mixed encodings such as Yes/Y/1/TRUE, ambiguous third states, unknown values.

## Diagnostic Evidence
Check distributions, frequencies, samples, pattern consistency, domain context, related columns, repeated occurrences, units, and whether an apparent anomaly is reproducible.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Map variants only when their truth semantics are unambiguous; preserve unknown/not-applicable as distinct from false.

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
