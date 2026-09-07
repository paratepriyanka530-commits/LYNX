---
knowledge_id: DQ-011
category: data_quality
topic: duplicate_records
version: 1.0
status: active
---

# Duplicate Records

## Definition
Two or more rows are exact or near-exact repetitions of the same recorded record.

## Semantic Indicators
Use column names, values, physical representation, uniqueness, neighboring columns, dataset context, and domain context as supporting evidence. No single indicator is proof.

## Important Distinctions
Exact duplicate rows differ from repeated legitimate events and duplicate entities.

## Quality Expectations
Values should be internally coherent, semantically plausible, consistently represented, and compatible with relevant cross-column relationships.

## Possible Data Quality Problems
Missingness, invalid values, implausible values, sentinel values, inconsistent representations, typographical errors, parsing errors, mixed units, and cross-column contradictions may occur.

## Diagnostic Evidence
Check all columns, timestamps, IDs and ingestion context.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Exact duplicate removal may be safe only when repetition has no semantic meaning; otherwise flag.

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
